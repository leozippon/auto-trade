"""The vendor client and offset paging (data_sources/tushare/client.py)."""

import json
import os
import tempfile
import threading
import traceback
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

from autotrade.data_sources.tushare import client as tushare_client
from autotrade.data_sources.tushare import download, specs, versions

from .tushare_fixture import LakeTestCase


class RepeatingPagedClient:
    def query(self, api_name, params=None, fields="", retries=5):
        result_fields = fields.split(",") if fields else ["trade_date", "ts_code"]
        row = ["20200102" if field == "trade_date" else "000001.SZ" for field in result_fields]
        return tushare_client.ApiResult(result_fields, [row])


class EndlessPagedClient:
    """Serves a full page of distinct rows at every offset, refusing an offset
    past the cap the way the vendor does (code 50101)."""

    def __init__(self):
        self.offsets = []

    def query(self, api_name, params=None, fields="", retries=5):
        offset, limit = params["offset"], params["limit"]
        if offset > tushare_client.VENDOR_MAX_OFFSET:
            raise RuntimeError(f"{api_name} answered code 50101: offset不能大于{tushare_client.VENDOR_MAX_OFFSET}")
        self.offsets.append(offset)
        return tushare_client.ApiResult(["trade_date", "ts_code"], [["20261103", str(offset + index)] for index in range(limit)])


class _VendorStub(BaseHTTPRequestHandler):
    """A loopback stand-in for the vendor's HTTP API: every POST is recorded
    and answered with the next queued (status, JSON body)."""

    answers: ClassVar[list[tuple[int, dict]]] = []
    requests: ClassVar[list[tuple[str, dict]]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        type(self).requests.append((self.path, json.loads(self.rfile.read(length) or b"{}")))
        status, body = type(self).answers.pop(0)
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:
        return None


class TuShareClientTest(unittest.TestCase):
    """The relay (中转) is the default transport; the official service serves
    only the datasets that say so, each with its own token. Requests go over
    real HTTP to a loopback stub, never to the vendor."""

    def setUp(self) -> None:
        _VendorStub.answers, _VendorStub.requests = [], []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _VendorStub)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.endpoint = f"http://127.0.0.1:{self.server.server_address[1]}/dataapi"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def test_official_client_pins_the_official_endpoint_and_its_own_token(self) -> None:
        env = {"TUSHARE_TOKEN": "relay-dummy", "TUSHARE_TOKEN_TMP": "official-dummy",
               "TUSHARE_RELAY_URL": "https://relay.example"}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, env, clear=True):
            client = tushare_client.official_client(Path(tmp), 0, 5)
        self.assertEqual(client.endpoint, tushare_client.OFFICIAL_TUSHARE_URL)
        self.assertEqual(client._token, "official-dummy")
        self.assertTrue(tushare_client.OFFICIAL_TUSHARE_URL.startswith("https://"))
        self.assertEqual(
            [n for n, s in specs.TEXT_SPECS.items() if not s.official], ["report_rc"]
        )

    def test_a_request_carries_the_sdk_payload_to_the_pinned_endpoint(self) -> None:
        _VendorStub.answers = [(200, {"code": 0, "msg": "", "data": {"fields": ["ts_code", "close"], "items": [["000001.SZ", 10.5]]}})]
        client = tushare_client.TuShareClient("secret", min_interval=0, timeout=5, endpoint=self.endpoint)
        result = client.query("daily", {"trade_date": "20261023"}, "ts_code,close")
        self.assertEqual((result.fields, result.items), (["ts_code", "close"], [["000001.SZ", 10.5]]))
        path, payload = _VendorStub.requests[0]
        self.assertEqual(path, "/dataapi/daily")
        self.assertEqual(payload, {
            "api_name": "daily",
            "token": "secret",
            "params": {"trade_date": "20261023", "ts_type_name": self.endpoint},
            "fields": "ts_code,close",
        })

    def test_an_http_error_status_raises_instead_of_answering_empty(self) -> None:
        # The SDK answered any status >= 400 with an empty frame, which a
        # writer cannot tell from a genuinely empty partition.
        _VendorStub.answers = [(502, {"error": "bad gateway"}), (500, {"error": "boom"})]
        client = tushare_client.TuShareClient("secret", min_interval=0, timeout=5, endpoint=self.endpoint)
        with patch.object(tushare_client.time, "sleep"), self.assertRaisesRegex(RuntimeError, "daily failed after 2 attempts: daily answered HTTP 500"):
            client.query("daily", {"trade_date": "20261023"}, "ts_code,close", retries=2)
        self.assertEqual(len(_VendorStub.requests), 2)

    def test_a_code_0_answer_without_its_data_block_raises_instead_of_answering_empty(self) -> None:
        _VendorStub.answers = [
            (200, {"code": 0, "msg": "", "data": None}),
            (200, {"code": 0, "msg": "", "data": {"fields": ["ts_code", "close"]}}),
        ]
        client = tushare_client.TuShareClient("secret", min_interval=0, timeout=5, endpoint=self.endpoint)
        with patch.object(tushare_client.time, "sleep"), self.assertRaisesRegex(RuntimeError, "daily answered code 0 without a data block"):
            client.query("daily", {"trade_date": "20261023"}, "ts_code,close", retries=2)
        # A genuinely empty answer still carries its fields and an empty list.
        _VendorStub.answers = [(200, {"code": 0, "msg": "", "data": {"fields": ["ts_code", "close"], "items": []}})]
        self.assertEqual(client.query("daily", {"trade_date": "20261024"}, "ts_code,close").items, [])

    def test_an_answer_without_a_requested_field_is_defective(self) -> None:
        _VendorStub.answers = [(200, {"code": 0, "msg": "", "data": {"fields": ["ts_code"], "items": [["000001.SZ"]]}})]
        client = tushare_client.TuShareClient("secret", min_interval=0, timeout=5, endpoint=self.endpoint)
        with self.assertRaisesRegex(versions.DefectiveAnswer, r"without requested fields \['close'\]"):
            client.query("daily", {"trade_date": "20261023"}, "ts_code,close")
        # Casing alone is not a defect: the macro tier maps vendor casing.
        _VendorStub.answers = [(200, {"code": 0, "msg": "", "data": {"fields": ["MONTH", "PMI"], "items": [["202609", 49.8]]}})]
        self.assertEqual(client.query("cn_pmi", {}, "month,pmi").fields, ["MONTH", "PMI"])

    def test_each_token_is_read_from_its_own_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            env_file = Path(tmp, ".env")
            env_file.write_text("TUSHARE_TOKEN_TMP=official-dummy\nTUSHARE_TOKEN=relay-dummy\n", encoding="utf-8")
            self.assertEqual(tushare_client.load_token(Path(tmp)), "relay-dummy")
            self.assertEqual(tushare_client.load_token(Path(tmp), tushare_client.OFFICIAL_TOKEN_ENV), "official-dummy")
            env_file.write_text("TUSHARE_TOKEN=relay-dummy\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "^TUSHARE_TOKEN_TMP is not set") as caught:
                tushare_client.load_token(Path(tmp), tushare_client.OFFICIAL_TOKEN_ENV)
            self.assertNotIn("relay-dummy", str(caught.exception))

    def test_token_never_reaches_an_error_or_its_traceback(self) -> None:
        token = "official-dummy-5f3a"
        _VendorStub.answers = [(500, {"msg": f"vendor echoed {token} for anns_d"})]
        client = tushare_client.TuShareClient(token, min_interval=0, timeout=5, endpoint=self.endpoint)
        with self.assertRaises(RuntimeError) as caught:
            client.query("anns_d", {"ann_date": "20230301"}, retries=1)
        rendered = "".join(traceback.format_exception(caught.exception))
        self.assertIn("vendor echoed <token> for anns_d", rendered)
        self.assertNotIn(token, rendered)

    def test_the_relay_is_the_default_endpoint_and_the_environment_overrides_it(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(tushare_client.TuShareClient("secret", timeout=17).endpoint, tushare_client.DEFAULT_TUSHARE_RELAY_URL)
        with patch.dict(os.environ, {"TUSHARE_RELAY_URL": "https://relay.example/"}):
            self.assertEqual(tushare_client.TuShareClient("secret").endpoint, "https://relay.example")

    def test_timeout_must_be_positive_integer(self) -> None:
        for timeout in (0, -1, True, 1.5):
            with self.subTest(timeout=timeout), self.assertRaisesRegex(ValueError, "positive integer"):
                tushare_client.TuShareClient("secret", timeout=timeout)  # type: ignore[arg-type]

    def test_download_cli_keeps_real_sdk_timeout_option(self) -> None:
        args = download.build_parser().parse_args(["download", "--tier", "daily", "--timeout-seconds", "23"])
        self.assertEqual(args.timeout_seconds, 23)


class QueryPagedTest(LakeTestCase):
    def test_query_paged_rejects_repeated_full_pages(self):
        with self.assertRaisesRegex(RuntimeError, "returned a repeated page"):
            tushare_client.query_paged(RepeatingPagedClient(), "daily", {"trade_date": "20200102"}, "trade_date,ts_code", page_limit=1)

    def test_query_paged_never_requests_past_the_vendor_offset_cap(self):
        # A day of 105,452 rows needs offset 104000 at 8,000 rows a page; the
        # vendor refuses it, and five retries cannot change that.
        client = EndlessPagedClient()
        with self.assertRaisesRegex(RuntimeError, "runs past the vendor's offset cap 100000"):
            tushare_client.query_paged(client, "top_list", {"trade_date": "20261103"}, "trade_date,ts_code", page_limit=8000)
        self.assertEqual(client.offsets, list(range(0, 96001, 8000)))


if __name__ == "__main__":
    unittest.main()
