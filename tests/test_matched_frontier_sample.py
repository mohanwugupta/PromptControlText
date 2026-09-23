import unittest
from unittest.mock import Mock,PropertyMock,patch
from types import SimpleNamespace
from experiments.run_matched_frontier_sample import SingleAttemptVLLM
from models.vllm_client import VLLMClient

class MatchedTests(unittest.TestCase):
    def test_one_call_preserves_messages_and_no_sdk_retry(self):
        sdk=Mock();sdk.with_options.return_value=sdk
        sdk.chat.completions.create.return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='Answer'),finish_reason='stop')],usage=None)
        with patch.object(VLLMClient,'client',new_callable=PropertyMock,return_value=sdk):
            client=SingleAttemptVLLM('historical',max_retries=1,enable_cache=False)
            text,meta=client.generate('Native\n\nController','User',max_tokens=512)
        self.assertEqual(text,'Answer')
        sdk.with_options.assert_called_once_with(max_retries=0)
        sdk.chat.completions.create.assert_called_once()
        self.assertEqual(sdk.chat.completions.create.call_args.kwargs['messages'],[{'role':'system','content':'Native\n\nController'},{'role':'user','content':'User'}])

if __name__=='__main__':unittest.main()
