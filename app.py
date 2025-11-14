# app.py

# -*- coding: utf-8 -*-

"""
Simple KV Proxy (бронебойный):

- Большие: LCP→restore, затем чат строго в этот же слот, потом save+meta.
- Малые: свободный/старый слот, без restore и без дискового save/meta.
- Пин slota дублируется в root/options/query (через клиента).

Дополнительно:

- acquire_for_request обёрнут в таймаут, чтобы не висеть бесконечно, если слот не отпускается.
- Для stream:
    * чтение из llama.cpp идёт в отдельной фоновой задаче (reader);
    * reader пушит чанки в asyncio.Queue;
    * в своём finally reader всегда делает save_after + write_meta + release(g),
      и кладёт в очередь sentinel None;
    * StreamingResponse читает из очереди и никак не влияет на release слота.
"""

import asyncio
import time
import logging
from typing import List, Dict, AsyncGenerator, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

from config import (
    BACKENDS,
    BIG_THRESHOLD_WORDS,
    LCP_TH,
    MODEL_ID,
    WORDS_PER_BLOCK,
    PORT,
)
import hashing as hs
from llama_client import LlamaClient
from slot_manager import SlotManager, GSlot

log = logging.getLogger(__name__)

ACQUIRE_TIMEOUT = 300.0
STREAM_QUEUE_SIZE = 16

app = FastAPI(title="Simple KV Proxy")


@app.on_event("startup")
async def startup():
    clients = [LlamaClient(be["url"]) for be in BACKENDS]
    sm = SlotManager()
    sm.set_clients(clients)
    app.state.clients = clients
    app.state.sm = sm
    log.info("app_start n_backends=%d port=%d", len(BACKENDS), PORT)


@app.on_event("shutdown")
async def shutdown():
    clients: List[LlamaClient] = getattr(app.state, "clients", [])
    if clients:
        await asyncio.gather(*(c.close() for c in clients))


@app.get("/v1/models")
async def models():
    return {"data": [{"id": MODEL_ID}]}


async def start_stream_task(
    resp: httpx.Response,
    g: GSlot,
    key: str,
    prefix: str,
    blocks: List[str],
    model_id: str,
    sm: SlotManager,
) -> AsyncGenerator[bytes, None]:
    queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=STREAM_QUEUE_SIZE)

    async def reader():
        try:
            log.info("stream_reader_start g=%s key=%s", g, key[:16])
            async for chunk in resp.aiter_raw():
                if not chunk:
                    continue
                try:
                    await queue.put(chunk)
                except asyncio.CancelledError:
                    log.warning("stream_reader_cancelled_put g=%s key=%s", g, key[:16])
                    raise
        except asyncio.CancelledError:
            log.warning("stream_reader_cancelled g=%s key=%s", g, key[:16])
            raise
        except Exception as e:
            log.exception("stream_reader_error g=%s key=%s: %s", g, key[:16], e)
        finally:
            try:
                await resp.aclose()
            except Exception:
                pass
            ok = False
            try:
                ok = await sm.save_after(g, key)
            except Exception as e:
                log.warning("save_after_exception g=%s key=%s: %s", g, key[:16], e)
            try:
                hs.write_meta(key, prefix, blocks, WORDS_PER_BLOCK, model_id)
            except Exception as e:
                log.warning("write_meta_exception key=%s: %s", key[:16], e)
            sm.release(g)
            log.info("stream_reader_done g=%s key=%s saved=%s", g, key[:16], ok)
            try:
                await queue.put(None)
            except Exception:
                pass

    asyncio.create_task(reader())

    async def gen() -> AsyncGenerator[bytes, None]:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    return gen()


@app.post("/v1/chat/completions")
async def chat(req: Request):
    sm: SlotManager = app.state.sm
    clients: List[LlamaClient] = app.state.clients

    t0 = time.time()
    data = await req.json()

    messages: List[Dict] = data.get("messages") or []
    stream = bool(data.get("stream", False))
    client_model = data.get("model") or MODEL_ID

    # model_id берём у первого backend'а
    backend_model_id = await clients[0].get_model_id()

    prefix = hs.raw_prefix(messages)
    full_for_key = backend_model_id + "\n" + prefix
    key = hs.prefix_key_sha256(full_for_key)
    blocks = hs.block_hashes_from_text(prefix, WORDS_PER_BLOCK)
    n_words = len(hs.words_from_text(prefix))
    is_big = n_words > BIG_THRESHOLD_WORDS

    restore_key: Optional[str] = None
    if is_big:
        cand = hs.find_best_restore_candidate(
            blocks,
            WORDS_PER_BLOCK,
            LCP_TH,
            backend_model_id,
        )
        if cand:
            restore_key, ratio = cand
            log.info(
                "restore_candidate basename=%s ratio=%.3f",
                restore_key[:16],
                ratio,
            )
        else:
            log.info("restore_candidate none")
    else:
        log.info(
            "small_request n_words=%d threshold=%d",
            n_words,
            BIG_THRESHOLD_WORDS,
        )

    log.info(
        "before_acquire is_big=%s restore_key=%s",
        is_big,
        restore_key[:16] if restore_key else None,
    )

    try:
        g, lock, restored = await asyncio.wait_for(
            sm.acquire_for_request(restore_key if is_big else None),
            timeout=ACQUIRE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        log.error(
            "acquire_timeout is_big=%s restore_key=%s",
            is_big,
            restore_key[:16] if restore_key else None,
        )
        return JSONResponse(
            {"error": "all slots busy, please retry later"},
            status_code=503,
        )

    log.info("after_acquire g=%s restored=%s", g, restored)

    be_id, slot_id = g
    client = clients[be_id]

    body = dict(data)
    body["model"] = client_model
    body["cache_prompt"] = bool(is_big)
    body["n_keep"] = -1

    opts = dict(body.get("options") or {})
    opts["slot_id"] = slot_id
    opts["id_slot"] = slot_id
    opts["n_keep"] = -1
    opts["cache_prompt"] = bool(is_big)
    body["options"] = opts

    log.info(
        "dispatch be=%d slot=%d is_big=%s (restore_target=%s restored=%s model_id=%s)",
        be_id,
        slot_id,
        is_big,
        restore_key[:16] if restore_key else None,
        restored,
        backend_model_id,
    )

    try:
        if stream:
            resp = await client.chat_completions(
                body,
                slot_id=slot_id,
                stream=True,
            )
            if resp.status_code != 200:
                err_txt = await resp.aread()
                await resp.aclose()
                sm.release(g)
                return JSONResponse(
                    {"error": err_txt.decode("utf-8", "ignore")},
                    status_code=resp.status_code,
                )

            gen = await start_stream_task(
                resp,
                g,
                key,
                prefix,
                blocks,
                backend_model_id,
                sm,
            )

            headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
            return StreamingResponse(
                gen,
                media_type="text/event-stream",
                headers=headers,
            )

        else:
            out = await client.chat_completions(
                body,
                slot_id=slot_id,
                stream=False,
            )
            if not isinstance(out, dict):
                sm.release(g)
                return JSONResponse(
                    {"error": "provider non-JSON body"},
                    status_code=502,
                )

            ok = False
            try:
                if is_big:
                    ok = await sm.save_after(g, key)
                    hs.write_meta(
                        key,
                        prefix,
                        blocks,
                        WORDS_PER_BLOCK,
                        backend_model_id,
                    )
            finally:
                sm.release(g)

            log.info(
                "json_done g=%s key=%s saved=%s is_big=%s dur_ms=%d",
                g,
                key[:16],
                ok,
                is_big,
                int((time.time() - t0) * 1000),
            )
            return JSONResponse(content=out, status_code=200)

    except Exception as e:
        sm.release(g)
        log.exception("chat_error g=%s key=%s: %s", g, key[:16], e)
        return JSONResponse({"error": str(e)}, status_code=500)
