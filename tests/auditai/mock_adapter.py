#!/usr/bin/env python3
"""Minimal mock HTTP target for guerrilla AuditAI runs.

Returns empty contexts so faithfulness uses dataset case.contexts.
Replace with real app endpoint before end-to-end audits.
"""
from __future__ import annotations
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

SEED = "A Retrieval Augmented Generation (RAG) chatbot that answers questions about Vietnam's 2019 Labor Code (Bộ luật Lao động 2019) , built with a hybrid retrieval pipeline, query rewriting, semantic routing, and a self correcting retrieval loop (Self RAG)."
ATTACK = re.compile(
    r"(ignore all previous|system prompt|api key|bỏ qua mọi|tiết lộ)",
    re.I,
)

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        q = str(body.get("question") or "")
        if ATTACK.search(q):
            ans = "I cannot reveal system prompts or secrets."
        elif SEED:
            ans = "Based on available docs: " + SEED[:500]
        else:
            ans = f"(mock) no seed docs for: {q[:80]}"
        # IMPORTANT: empty contexts → AuditAI falls back to dataset contexts
        raw = json.dumps({"answer": ans, "contexts": []}, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

print("mock adapter http://127.0.0.1:18080/chat (empty response contexts)")
HTTPServer(("127.0.0.1", 18080), H).serve_forever()
