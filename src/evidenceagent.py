import autogen

class EvidenceAutoAgent:
    def __init__(self,model,api_key,api_base_url,version,agent_max_round=12):
        self.llm_config = {
            "config_list": [
                {
                    "model": model,
                    "api_type": "azure",
                    "api_key": api_key,
                    "base_url": api_base_url,
                    "api_version": version
                }
            ],
        }

        self.assistant = autogen.AssistantAgent(
            name="Document Evidence Extractor",
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            system_message="""
                You are a Document Evidence Extractor. Your job is to find the right document evidence text that matches each of the given Intervention, Comparator, and Outcome.
                #Format:
                You should always return information in the following format FOLLOWED BY WORD TERMINATE (in caps):
                - Original evidences, comma-separated: [List of unchanged original evidences from the document]
            """,
            llm_config=self.llm_config,
            human_input_mode="NEVER",
        )

        self.suggester = autogen.AssistantAgent(
            name="Additional Info Suggester",
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            system_message="""
                You are a Additional Info Suggester. Your job is to help the assistant find additional information for given document evidence which supports the evidences extracted for Intervention, Comparator, Outcome.
                #Tasks:
                - Extract detailed descriptions or data related to the specified outcomes, statistical analyses, mechanisms of action, comparative efficacy and safety, and discussion and critique sections.
                - Extract direct quotations from the document that support the effectiveness or drawbacks of the intervention as compared to the comparator, with specific emphasis on the impact on the outcome.
                You must end your response BY WORD TERMINATE (all in caps).
            """,
            llm_config=self.llm_config,
            human_input_mode="NEVER",
        )

        self.user_proxy = autogen.UserProxyAgent(
            name="Manager",
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            system_message="""
                You are a manager. Your job is to manage invoking the extractor, suggester, aggregator.
            """,
            llm_config=self.llm_config,
            human_input_mode="NEVER",
        )

        self.aggregator = autogen.AssistantAgent(
            name="Aggregator",
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            system_message="""
                You are an Aggregator. Your job is to aggregate the full Original Evidence extracted and the full additional information suggested for Intervention, Comparator, Outcome.
                You must return in the following format ending with explicit word -TERMINATE (in caps):
                    - [LIST OF ORIGINAL EVIDENCES GATHERED BY ASSISTANT]
                    - [LIST OF ADDITIONAL INFORMATION GATHERED BY SUGGESTER]
                    TERMINATE
            """,
            llm_config=self.llm_config,
            human_input_mode="NEVER",
        )

        self.groupchat = autogen.GroupChat(agents=[self.user_proxy, self.assistant, self.suggester, self.aggregator], messages=[], max_round=agent_max_round)
        self.manager = autogen.GroupChatManager(groupchat=self.groupchat, llm_config=self.llm_config)

    def submit_task(self, clinical_trial_doc,ico_data):
        assistant_msg = f"Your task is to find matching document evidence sentences from the clinical trial document for each of Intervention, Comparator, Outcome.\nClinical trial document: {clinical_trial_doc}\nICO Details: {ico_data}"
        suggester_msg = f"Your task is to find additional document information from Clinical Trail document which supports the document evidence sentences extracted so far for Intervention,Comparator,Outcome.\nClinical trial document: {clinical_trial_doc}\n"
        aggregator_msg = f"Your task is to aggregate the document evidence sentences and additional information collected for each of Intervention,Comparator,Outcome\n"

        chat_result = self.user_proxy.initiate_chats(
            [
                {"recipient": self.assistant, "message": assistant_msg, "silent": False},
                {"recipient": self.suggester, "message": suggester_msg},
                {"recipient": self.aggregator, "message": aggregator_msg}
            ]
        )

        last_but_one_chat_content = chat_result[2].chat_history[-1]['content']
        lines = last_but_one_chat_content.split('\n')
        if 'TERMINATE' in lines[-1]:
            lines = lines[:-1]
        cleaned_content = '\n'.join(lines)
        return cleaned_content

