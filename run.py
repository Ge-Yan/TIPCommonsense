import sys
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM, pipeline
import openai
import time
import torch
from utils.data_utils import write_to_csv
import os
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
import hydra
import logging
import gc
from omegaconf import OmegaConf
from datetime import datetime
import wandb
from utils.str_utils import clean_sentence
from utils.str_utils import extract_content_before_substring
from hydra.core.hydra_config import HydraConfig

logger = logging.getLogger(__name__)
from hydra import utils
from utils.eval import eval
from utils.data_utils import read_dataset


class ResponseGenerator:
    def __init__(self, args, generate_answer: bool):
        if generate_answer:
            self.model_name = args.model_setting.model_name
            self.model_setting = args.model_setting

        else:
            self.model_name = args.knowledge_model_setting.model_name
            self.model_setting = args.knowledge_model_setting

        logger.info("Model name: " + self.model_name)
        logger.info("Set batch size to " + str(self.model_setting.batch_size))


		self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True, padding_side='left',
													   cache_dir="/home/iiserver32/.cache/huggingface/hub/")
	  

		self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto",
														  torch_dtype=torch.float16,
														  cache_dir="/home/iiserver32/.cache/huggingface/hub/")

		self.text_generator = pipeline(
			"text-generation",
			model=self.model,
			tokenizer=self.tokenizer,
			torch_dtype=torch.float16,
			device_map="auto"
		)
		self.text_generator.tokenizer.pad_token_id = self.text_generator.model.config.eos_token_id


    def generate_response(self, prompt_content):
        response = ""
        response = self.text_generator(
                prompt_content,
                batch_size=self.model_setting.batch_size,
                do_sample=self.model_setting.do_sample,
                top_k=self.model_setting.top_k,
                temperature=self.model_setting.temperature,
                top_p=self.model_setting.top_p,
                num_return_sequences=1,
                return_full_text=self.model_setting.return_full_text,
                eos_token_id=self.tokenizer.eos_token_id,
                max_length=self.model_setting.max_length,
               
            )
        return response


def prompting_model(prompts, model, model_name, desc):
    answers = []
    if "gpt" not in model_name:
        data = Dataset.from_list(prompts)
        responses = model.generate_response(KeyDataset(data, "content"))

        logger.info("Start generating responses...")
        start_time = time.time()
        for response, prompt in tqdm(zip(responses, prompts), total=len(data), desc=desc):
            response = clean_sentence(response[0]["generated_text"])
            original_response = response
            processed_response = extract_content_before_substring(response, "Given")
            
            answers.append({
                'id': prompt['id'],
                'answer': processed_response,
                'prompt': prompt["content"]
            })
        end_time = time.time()
        runtime_minutes = (end_time - start_time) / 60

        logger.info("Finished generating responses...")
        logger.info(f"Total runtime: {runtime_minutes:.2f} minutes")
    else:
        for prompt in tqdm(prompts, desc=desc):
            answer_content = model.generate_response(prompt["content"])
            answers.append({
                'id': prompt['id'],
                'answer': answer_content,
                'prompt': prompt["content"]
            })
    return answers


def make_prompts(dataset, knowledge_generator, args):
    prompts = []
    knowledge_prompts = []
    if args.with_concepts:
        knowledge_prompt_format = "Given the nonsensical statement {statement} and some concepts in this statement: {concepts}. Generate commonsense knowledge of these concepts that is related to the statement:"

        for data in dataset:

            if args.random_knowledge:
                knowledge_prompt = "please generate knowledge based on the sentence." + data[
                    'FalseSent'] + "."
            else:
                knowledge_prompt = knowledge_prompt_format.format(statement="'" + data['FalseSent'] + "'",
                                                                  concepts=", ".join(data["concepts"]))
            knowledge_prompts.append({
                'id': data['id'],
                'content': knowledge_prompt
            })
        knowledge_results = prompting_model(knowledge_prompts, knowledge_generator,
                                            args.knowledge_model_setting.model_name,
                                            desc="Generating knowledge")
        if args.prompts_style == "COT":
            prompt_format = "Given the commonsense knowledge: {knowledge} Please explain why this statement defies common sense based on these knowledge. Statement: {statement}. Think step by step and provide a detailed reasoning process. The explanation is:"

     
        else:

            prompt_format = "Given the knowledge: {knowledge} Please explain why this statement defies commonsense. Statement: {statement}."

        for knowledge, data in zip(knowledge_results, dataset):
            if args.random_knowledge:
                knowledge_content = ". ".join(knowledge["answer"].split(".")[:2])
            else:
                knowledge_content = ". ".join(knowledge["answer"].split(".")[:2]).strip()
            if not knowledge_content.endswith("."):
                knowledge_content = knowledge_content + "."
       
            if data['FalseSent'].endswith('.'):
                data['FalseSent'].rstrip('.')

            knowledge_content = clean_sentence(extract_content_before_substring(knowledge_content, "Given"))
            prompts.append({
                'id': knowledge["id"],
                'content': prompt_format.format(knowledge=knowledge_content, statement=data['FalseSent'])
            })

    else:
       
        args.num_concept = 0
        if args.prompts_style == "COT":
            prompt_format = "Please explain why this statement defies commonsense. Statement: {statement}. Think step by step and provide a detailed reasoning process. The explanation is:"
        elif args.prompts_style == "APE":
            prompt_format = "Generate a coherent sentence that explains the input. Input: {statement}."
            
        else:
           
            prompt_format = "explain why the sentence defies commonsense. {statement}."


        for data in tqdm(dataset, "Generating prompts"):
            prompts.append({
                'id': data['id'],
                'content': prompt_format.format(statement=data['FalseSent'])
            })

    return prompts


def run(file: str, isDebug: bool):
    @hydra.main(config_path="conf", config_name="config")
    def main(cfg):
        log_dir = HydraConfig.get().run.dir

        log_file = os.path.join(log_dir, 'run.log')
        logging.basicConfig(filename=log_file,
                            level=logging.ERROR,
                            format='%(asctime)s %(levelname)s %(message)s')

        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

        sys.excepthook = handle_exception

        if cfg.do_eval_only:
            eval(cfg, isDebug=isDebug)
        else:

            ori_cwd = hydra.utils.get_original_cwd()
            cwd = os.getcwd()
            cfg.cwd = cwd
            args = cfg

            logger.info(cfg)
            logger.info("======================================================")

            logger.info("Model name:" + args.model_setting.model_name)

            if args.with_concepts:
                logger.info("Knowledge_model name:" + args.knowledge_model_setting.model_name)
                logger.info("Concept number:" + str(args.num_concept))
                logger.info("With concept:" + str(args.with_concepts))
                logger.info("Random knowledge:" + str(args.random_knowledge))

            logger.info("Prompt style:" + str(args.prompts_style))
            logger.info("Task:" + str(args.task))

            logger.info("======================================================")
            file_name = file

            number_concept_in_file = args.num_concept
            if not args.with_concepts:
                args.knowledge_model_setting.model_name = "without"
            name = ""
            if args.knowledge_model_setting.model_name != "without":
               
                name = args.model_setting.model_name.split("/")[1] + "_" + \
                args.knowledge_model_setting.model_name.split("/")[1] + "_" + str(args.num_concept)
            else:
                
                name = args.model_setting.model_name.split("/")[1]
                args.num_concept = 0
            if not isDebug:
                wandb.login()
                wandb.init(
                    project=args.wandb_project_name,
                    name=name,

                    config={
                        "knowledge_model": args.knowledge_model_setting.model_name,
                        "model": args.model_setting.model_name,
                        "num_concept": args.num_concept
                    })

            concepts_data = read_dataset(os.path.join(ori_cwd, args.concepts_path), read_concept=True,
                                         num_concept=number_concept_in_file)
            if args.tiny_data:
                concepts_data = concepts_data[:100]

            if args.with_concepts:
                knowledge_generator = ResponseGenerator(args, generate_answer=False)
                prompts = make_prompts(concepts_data, knowledge_generator, args)             
                knowledge_generator = None
                gc.collect()
                torch.cuda.empty_cache()
                answer_generator = ResponseGenerator(args, generate_answer=True)
            else:
                knowledge_generator = None
                prompts = make_prompts(concepts_data, knowledge_generator, args)
                answer_generator = ResponseGenerator(args, generate_answer=True)

            answers = prompting_model(prompts, answer_generator, args.model_setting.model_name,
                                      desc="Generating explanations")

            write_to_csv(file_name, answers, args.output_path)
            args.pred_file = os.path.join(cwd, file_name)
            answer_generator = None
            gc.collect()
            torch.cuda.empty_cache()
            eval(args, isDebug=isDebug)

    main()


if __name__ == "__main__":

    config_path = "/data/ge_yan/Nedo/CSEG/conf/config.yaml"
    config = OmegaConf.load(config_path)
    now = datetime.now().strftime("%m-%d_%H-%M-%S")
    dir_name = ''
    if len(sys.argv) > 1:
        if "with_concepts=True" in sys.argv:
            config["model_setting"]["model_name"] = sys.argv[1].split("=")[1]
            config["knowledge_model_setting"]["model_name"] = sys.argv[2].split("=")[1]
            config["prompts_style"] = sys.argv[3].split("=")[1]
            config["with_concepts"] = True
        else:
            config["model_setting"]["model_name"] = sys.argv[1].split("=")[1]
            config["prompts_style"] = sys.argv[2].split("=")[1]
            config["with_concepts"] = False
    if config["with_concepts"]:
        if "gpt" in config["model_setting"]["model_name"]:
            dir_name = config["model_setting"]["model_name"] + "_" + \
                       config["knowledge_model_setting"]["model_name"].split("/")[1] + "_" + \
                       os.path.splitext(config["concepts_path"].split("/")[-1])[0] + "_" + config[
                           "prompts_style"] + "_" + \
                       config["task"]
        else:
            dir_name = config["model_setting"]["model_name"].split("/")[1] + "_" + \
                       config["knowledge_model_setting"]["model_name"].split("/")[1] + "_" + \
                       os.path.splitext(config["concepts_path"].split("/")[-1])[0] + "_" + config["prompts_style"] + "_" + \
                       config["task"]
        if config["random_knowledge"]:
            dir_name = dir_name + "_" + "random"
    else:
        if "gpt" in config["model_setting"]["model_name"]:
            dir_name = config["model_setting"]["model_name"] + "_" + config["prompts_style"] + "_" + \
                       config["task"]
        else:
            dir_name = config["model_setting"]["model_name"].split("/")[1] + "_" + config["prompts_style"] + "_" + config[
            "task"]

    isDebug = True if sys.gettrace() else False
    if isDebug:
        arg_dir = ''
    else:
        arg_dir = ''
    sys.argv.append(arg_dir)

    file_name = dir_name + ".csv"
    run(file=file_name, isDebug=isDebug)
