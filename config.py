import argparse


def config():
    parser = argparse.ArgumentParser(description='')

    # Saving parameters
    parser.add_argument('--model_name', type=str, help='Name of model used to test.')
    parser.add_argument('--knowledge_model_name', type=str, help='Name of model used to generate knowledge.')

    # Decoding Parameters
    parser.add_argument('--temperature', type=float, default=0, help='Decoding temperature.')
    parser.add_argument('--top_p', type=float, default=1.0, help='Decoding top-p.')
    parser.add_argument('--max_events_length', type=int, default=1900, help='Max input article length.')
    parser.add_argument('--max_batch_size', type=int, default=1, help='Max batch size.')
    parser.add_argument('--max_new_decoding_tokens', type=int, default=2, help='Max new tokens for decoding output.')

    # Experiment Parameters
    parser.add_argument("--do_eval", action="store_true", help="Do experiments.")
    parser.add_argument('--prompt_style', type=str, default='mcq',
                        help='Prompt style used for McTACO dataset. Either mcq or qa.')

    parser.add_argument('--with_concepts', action="store_true", help="Do experiments with concepts.")

    parser.add_argument('--add_special_tokens', action="store_true", help="Add cls for model input.")
    parser.add_argument('--remove_punctuation', action="store_true", help="Remove punctuation for model input.")
    parser.add_argument('--num_concept', type=int, default=2, help="Number of concepts.")



    # LLM ability test
    parser.add_argument("--do_few_shot_learning", action="store_true", help="Do ICL prompting.")

    # Path parameters
    parser.add_argument('--data_path', type=str, help='Path for model inputs.')
    parser.add_argument('--concepts_path', type=str, help='Path for concepts.')

    parser.add_argument('--output_path', type=str, help='Path for model outputs.')

    args = parser.parse_args()

    return args

