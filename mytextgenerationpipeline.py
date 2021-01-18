import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'PPLM')
# from PPLM import run_pplm, pplm_classification_head

# from run_pplm import run_pplm_example
from run_pplm import get_bag_of_words_indices, set_generic_model_params, get_classifier, generate_text_pplm
from run_pplm import DISCRIMINATOR_MODELS_PARAMS, PPLM_BOW_DISCRIM, PPLM_BOW, PPLM_DISCRIM
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import GPT2Tokenizer
import torch
import numpy as np

# model = GPT2LMHeadModel.from_pretrained(
#     "C:/Users/Admin/simple_text_generation/mygpt2-medium",
#     output_hidden_states=True)
#

QUIET = 0
REGULAR = 1
VERBOSE = 2
VERY_VERBOSE = 3
VERBOSITY_LEVELS = {
    'quiet': QUIET,
    'regular': REGULAR,
    'verbose': VERBOSE,
    'very_verbose': VERY_VERBOSE,
}


def full_text_generation(
        model,
        tokenizer,
        context=None,
        num_samples=1,
        device="cuda",
        bag_of_words=None,
        discrim=None,
        class_label=None,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        verbosity_level=REGULAR,
        **kwargs
):
    classifier, class_id = get_classifier(
        discrim,
        class_label,
        device
    )

    unpert_gen_tok_text, _, _ = generate_text_pplm(
        model=model,
        tokenizer=tokenizer,
        context=context,
        device=device,
        length=length,
        sample=sample,
        perturb=False,
        verbosity_level=verbosity_level
    )

    bow_indices = []
    if bag_of_words:
        bow_indices = get_bag_of_words_indices(bag_of_words.split(";"),
                                               tokenizer)

    if bag_of_words and classifier:
        loss_type = PPLM_BOW_DISCRIM
        if verbosity_level >= REGULAR:
            print("Both PPLM-BoW and PPLM-Discrim are on. "
                  "This is not optimized.")

    elif bag_of_words:
        loss_type = PPLM_BOW
        if verbosity_level >= REGULAR:
            print("Using PPLM-BoW")

    elif classifier is not None:
        loss_type = PPLM_DISCRIM
        if verbosity_level >= REGULAR:
            print("Using PPLM-Discrim")

    else:
        # TODO: Try to prepare something better
        if device == 'cuda':
            torch.cuda.empty_cache()
        return unpert_gen_tok_text, None, None, None # pert_gen_tok_texts, discrim_losses, losses_in_time
        raise Exception("Specify either a bag of words or a discriminator")


    if device == 'cuda':
        torch.cuda.empty_cache()

    pert_gen_tok_texts = []
    discrim_losses = []
    losses_in_time = []

    for i in range(num_samples):
        pert_gen_tok_text, discrim_loss, loss_in_time = generate_text_pplm(
            model=model,
            tokenizer=tokenizer,
            context=context,
            device=device,
            perturb=True,
            bow_indices=bow_indices,
            classifier=classifier,
            class_label=class_id,
            loss_type=loss_type,
            length=length,
            stepsize=stepsize,
            temperature=temperature,
            top_k=top_k,
            sample=sample,
            num_iterations=num_iterations,
            grad_length=grad_length,
            horizon_length=horizon_length,
            window_length=window_length,
            decay=decay,
            gamma=gamma,
            gm_scale=gm_scale,
            kl_scale=kl_scale,
            verbosity_level=verbosity_level
        )
        pert_gen_tok_texts.append(pert_gen_tok_text)
        if classifier is not None:
            discrim_losses.append(discrim_loss.data.cpu().numpy())
        losses_in_time.append(loss_in_time)

    if device == 'cuda':
        torch.cuda.empty_cache()

    return unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time

def  run_my_pplm(
        # pretrained_model="gpt2-medium",
        model,
        tokenizer,
        cond_text="",
        uncond=False,
        num_samples=1,
        bag_of_words=None,
        discrim=None,
        discrim_weights=None,
        discrim_meta=None,
        class_label=-1,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        seed=0,
        no_cuda=False,
        colorama=False,
        verbosity='regular'
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if bag_of_words is 'None':
        bag_of_words = None


    # set verbosiry
    verbosity_level = VERBOSITY_LEVELS.get(verbosity.lower(), REGULAR)

    if discrim == 'generic':
        set_generic_model_params(discrim_weights, discrim_meta)

    if discrim is not None:
        discriminator_pretrained_model = DISCRIMINATOR_MODELS_PARAMS[discrim][
            "pretrained_model"
        ]
        if pretrained_model != discriminator_pretrained_model:
            pretrained_model = discriminator_pretrained_model
            if verbosity_level >= REGULAR:
                print("discrim = {}, pretrained_model set "
                "to discriminator's = {}".format(discrim, pretrained_model))

    # # load pretrained model
    # model = GPT2LMHeadModel.from_pretrained(
    #     pretrained_model,
    #     output_hidden_states=True
    # )

    # tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
    # figure out conditioning text
    if uncond:
        tokenized_cond_text = tokenizer.encode(
            [tokenizer.bos_token],
            add_special_tokens=False
        )
    else:
        raw_text = cond_text
        while not raw_text:
            print("Did you forget to add `--cond_text`? ")
            raw_text = input("Model prompt >>> ")
        tokenized_cond_text = tokenizer.encode(
            tokenizer.bos_token + raw_text,
            add_special_tokens=False
        )


    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
    model.to(device)
    model.eval()

    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    unpert_gen_tok_text, pert_gen_tok_texts, _, _ = full_text_generation(
        model=model,
        tokenizer=tokenizer,
        context=tokenized_cond_text,
        device=device,
        num_samples=num_samples,
        bag_of_words=bag_of_words,
        discrim=discrim,
        class_label=class_label,
        length=length,
        stepsize=stepsize,
        temperature=temperature,
        top_k=top_k,
        sample=sample,
        num_iterations=num_iterations,
        grad_length=grad_length,
        horizon_length=horizon_length,
        window_length=window_length,
        decay=decay,
        gamma=gamma,
        gm_scale=gm_scale,
        kl_scale=kl_scale,
        verbosity_level=verbosity_level
    )

    # untokenize unperturbed text
    # unpert_gen_text = tokenizer.decode(unpert_gen_tok_text.tolist()[0])

    unpert_gen_text = tokenizer.decode(
        unpert_gen_tok_text.tolist()[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    if verbosity_level >= REGULAR:
        print("=" * 80)

    print("= Unperturbed generated text =")
    print(unpert_gen_text)

    generated_texts = []
    bow_word_ids = set()

    if bag_of_words and colorama:
        bow_indices = get_bag_of_words_indices(bag_of_words.split(";"),
                                               tokenizer)
        for single_bow_list in bow_indices:
            # filtering all words in the list composed of more than 1 token
            filtered = list(filter(lambda x: len(x) <= 1, single_bow_list))
            # w[0] because we are sure w has only 1 item because previous fitler
            bow_word_ids.update(w[0] for w in filtered)

    # iterate through the perturbed texts
    if pert_gen_tok_texts:
        for i, pert_gen_tok_text in enumerate(pert_gen_tok_texts):
            try:
                # untokenize unperturbed text
                if colorama:
                    import colorama

                    pert_gen_text = ''
                    for word_id in pert_gen_tok_text.tolist()[0]:
                        if word_id in bow_word_ids:
                            # pert_gen_text += '{}{}{}'.format(
                            #     colorama.Fore.RED,
                            #     tokenizer.decode([word_id],
                            #                     skip_special_tokens=True,
                            #                     clean_up_tokenization_spaces=False,),
                            #     colorama.Style.RESET_ALL
                            # )
                            pert_gen_text += '{}{}{}'.format(
                                '<span style="color:blue">',
                                tokenizer.decode([word_id],
                                                skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False,),
                                "</span>"
                            )
                        else:
                            pert_gen_text += tokenizer.decode([word_id],
                                                            skip_special_tokens=True,
                                                            clean_up_tokenization_spaces=False,)
                else:
                    # pert_gen_text = tokenizer.decode(pert_gen_tok_text.tolist()[0])
                    pert_gen_text = tokenizer.decode(
                        pert_gen_tok_text.tolist()[0],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                print([tokenizer.decode(bwid) for bwid in bow_word_ids])
                print("= Perturbed generated text {} =".format(i + 1))
                print(pert_gen_text)
                print()
            except:
                print("""ERROR in pert""")
                pass

            # keep the prefix, perturbed seq, original seq for each index
            # generated_texts.append(
            #     (unpert_gen_tok_text, tokenized_cond_text, pert_gen_tok_text)
            # )
            generated_texts.append(
                (unpert_gen_text, pert_gen_text)
            )
        else:
            generated_texts.append((unpert_gen_text, ""))
    return generated_texts





