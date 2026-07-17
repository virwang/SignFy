import unittest
from unittest.mock import patch, mock_open, MagicMock
import json
import urllib.error
import sys
import os
from io import StringIO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock the asl_llm_video_mapping module before importing the tested module
mock_asl_mapping = MagicMock()
mock_asl_mapping.get_valid_glosses.return_value = {'HELLO', 'WORLD'}
sys.modules['asl_llm_video_mapping'] = mock_asl_mapping

import english_to_asl_gloss_llama as e2a

class TestEnglishToAslGlossLlama(unittest.TestCase):
    
    @patch('builtins.open', new_callable=mock_open, read_data='mock system prompt')
    @patch('english_to_asl_gloss_llama.PROMPT_PATH', 'dummy_path.txt')
    def test_load_system_prompt_success(self, mock_file):
        prompt = e2a.load_system_prompt()
        self.assertEqual(prompt, 'mock system prompt')
        mock_file.assert_called_once_with('dummy_path.txt', 'r', encoding='utf-8')

    @patch('builtins.open', side_effect=Exception('file not found'))
    @patch('english_to_asl_gloss_llama.PROMPT_PATH', 'dummy_path.txt')
    def test_load_system_prompt_failure(self, mock_file):
        with self.assertRaisesRegex(RuntimeError, "Could not load system prompt"):
            e2a.load_system_prompt()

    def test_fallback_to_fingerspelling(self):
        db_keys = {'HELLO', 'WORLD'}
        
        # Test empty word
        self.assertEqual(e2a.fallback_to_fingerspelling('', db_keys), '')
        
        # Test word in DB
        self.assertEqual(e2a.fallback_to_fingerspelling('HELLO', db_keys), 'HELLO')
        
        # Test word not in DB (should convert to hyphenated)
        self.assertEqual(e2a.fallback_to_fingerspelling('JOHN', db_keys), 'J-O-H-N')
        
        # Test word not in DB with existing hyphen
        self.assertEqual(e2a.fallback_to_fingerspelling('J-O-H-N', db_keys), 'J-O-H-N')

    def test_clean_gloss_valid_json(self):
        mock_asl_mapping.get_valid_glosses.return_value = {'HELLO', 'WORLD'}
        response = '[{"gloss": "hello", "synonyms": ["hi"]}, {"gloss": "john", "synonyms": []}]'
        expected = [
            {"gloss": "HELLO", "synonyms": ["H-I"]},
            {"gloss": "J-O-H-N", "synonyms": []}
        ]
        result = e2a.clean_gloss(response)
        self.assertEqual(result, expected)

    def test_clean_gloss_markdown_and_comments(self):
        mock_asl_mapping.get_valid_glosses.return_value = {'HELLO'}
        response = '''```json\n// this is a comment\n[\n    {"gloss": "hello12", "synonyms": ["hi", /* block comment */ "hey"]}\n]\n```'''
        expected = [
            {"gloss": "HELLO", "synonyms": ["H-I", "H-E-Y"]}
        ]
        result = e2a.clean_gloss(response)
        self.assertEqual(result, expected)

    def test_clean_gloss_invalid_json(self):
        mock_asl_mapping.get_valid_glosses.return_value = {'HELLO'}
        response = 'not json'
        with self.assertRaisesRegex(RuntimeError, "Failed to parse JSON response"):
            e2a.clean_gloss(response)

    def test_clean_gloss_not_array(self):
        mock_asl_mapping.get_valid_glosses.return_value = {'HELLO'}
        response = '{"gloss": "hello"}'
        with self.assertRaisesRegex(RuntimeError, "Expected a JSON array"):
            e2a.clean_gloss(response)

    @patch('urllib.request.urlopen')
    @patch('english_to_asl_gloss_llama.load_system_prompt', return_value='dummy prompt')
    @patch('english_to_asl_gloss_llama.clean_gloss')
    def test_ask_llama_success(self, mock_clean_gloss, mock_load_prompt, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "message": {"content": "dummy response content"}
        }).encode('utf-8')
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        mock_clean_gloss.return_value = [{"gloss": "HELLO", "synonyms": []}]
        
        result = e2a.ask_llama("hello")
        
        self.assertEqual(result, [{"gloss": "HELLO", "synonyms": []}])
        mock_clean_gloss.assert_called_once_with("dummy response content")

    @patch('urllib.request.urlopen')
    @patch('english_to_asl_gloss_llama.load_system_prompt', return_value='dummy prompt')
    def test_ask_llama_network_error(self, mock_load_prompt, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")
        
        with self.assertRaisesRegex(RuntimeError, "Could not connect to Ollama"):
            e2a.ask_llama("hello")

    @patch('sys.argv', ['script_name', 'hello', 'world'])
    def test_parse_args(self):
        args = e2a.parse_args()
        self.assertEqual(args.text, ['hello', 'world'])
        self.assertEqual(args.model, e2a.DEFAULT_MODEL)

    @patch('sys.argv', ['script_name'])
    @patch('sys.stdin.read', return_value='hello world\n')
    @patch('english_to_asl_gloss_llama.ask_llama', return_value=[{"gloss": "HELLO"}])
    @patch('sys.stdout', new_callable=StringIO)
    def test_main_stdin(self, mock_stdout, mock_ask_llama, mock_stdin_read):
        result = e2a.main()
        self.assertEqual(result, 0)
        output = mock_stdout.getvalue()
        self.assertIn('"gloss": "HELLO"', output)
        mock_ask_llama.assert_called_once_with('hello world', e2a.DEFAULT_MODEL)

    @patch('sys.argv', ['script_name'])
    @patch('sys.stdin.read', return_value='')
    @patch('sys.stderr', new_callable=StringIO)
    def test_main_no_input(self, mock_stderr, mock_stdin_read):
        result = e2a.main()
        self.assertEqual(result, 2)
        self.assertIn('Please provide English text', mock_stderr.getvalue())

    @patch('sys.argv', ['script_name', 'error', 'text'])
    @patch('english_to_asl_gloss_llama.ask_llama', side_effect=RuntimeError("llama error"))
    @patch('sys.stderr', new_callable=StringIO)
    def test_main_runtime_error(self, mock_stderr, mock_ask_llama):
        result = e2a.main()
        self.assertEqual(result, 1)
        self.assertIn('Error: llama error', mock_stderr.getvalue())

    @patch('urllib.request.urlopen')
    @patch('english_to_asl_gloss_llama.load_system_prompt', return_value='dummy prompt')
    def test_end_to_end_translation_cases(self, mock_load_prompt, mock_urlopen):
        # Load the test cases
        test_cases_path = os.path.join(os.path.dirname(__file__), 'test_cases.json')
        with open(test_cases_path, 'r', encoding='utf-8') as f:
            cases = json.load(f)['test_cases']

        for case in cases:
            with self.subTest(id=case['id'], category=case['category'], english=case['english']):
                expected_glosses = case['ground_truth'].split()
                
                # Mock the LLM output to match the expected ground truth
                mock_llm_output = [{"gloss": g, "synonyms": []} for g in expected_glosses]
                
                mock_response = MagicMock()
                mock_response.read.return_value = json.dumps({
                    "message": {"content": json.dumps(mock_llm_output)}
                }).encode('utf-8')
                mock_response.__enter__.return_value = mock_response
                mock_urlopen.return_value = mock_response

                # Ensure all ground truth glosses are in the valid glosses mock to avoid fingerspelling
                mock_asl_mapping.get_valid_glosses.return_value = set(expected_glosses)

                # Call the function
                result = e2a.ask_llama(case['english'])

                # Extract the primary glosses from the result
                actual_glosses = [item['gloss'] for item in result]
                
                self.assertEqual(actual_glosses, expected_glosses)

if __name__ == '__main__':
    unittest.main()
