from langchain import hub
prompt = hub.pull("rlm/rag-prompt")
import pprint


# pprint.pprint(prompt.messages)