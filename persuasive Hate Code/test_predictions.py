import torch
from torch.utils.data import DataLoader
import config
import numpy as np
import pandas as pd
from ast import literal_eval
from transformers import RobertaTokenizer
from dataset import Multimodal_Data  # If using PBM
from roberta_dataset import Roberta_Data  # If using Roberta
from roberta_baseline import build_baseline  # Import your model construction function

def predict(opt, model, test_loader):
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            cap = batch['cap_tokens'].long().cuda()
            mask = batch['mask'].cuda()
            feat = None

            if opt.MODEL == 'pbm':
                mask_pos = batch['mask_pos'].cuda()
                logits = model(cap, mask, mask_pos, feat)
                # If needed, you can apply post-processing to logits here
            elif opt.MODEL == 'roberta':
                if opt.UNIMODAL == False:
                    feat = batch['feat'].cuda()
                logits = model(cap, mask, feat)
                # If needed, you can apply post-processing to logits here

            predictions = torch.sigmoid(logits)
            all_predictions.append(predictions.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    return all_predictions

if __name__ == '__main__':
    opt = config.parse_opt()
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    constructor = 'build_baseline'
    if opt.MODEL == 'pbm':
        from dataset import Multimodal_Data
        import baseline
        test_set = Multimodal_Data(opt, tokenizer, opt.DATASET, 'test')
        label_list = [test_set.label_mapping_id[i] for i in test_set.label_mapping_word.keys()]
        model = getattr(baseline, constructor)(opt, label_list).cuda()
    else:
        from roberta_dataset import Roberta_Data
        import roberta_baseline
        test_set = Roberta_Data(opt, tokenizer, opt.DATASET, 'test')
        model = getattr(roberta_baseline, constructor)(opt).cuda()

    test_loader = DataLoader(test_set, opt.BATCH_SIZE, shuffle=False, num_workers=1)

    # Load the trained model
    model_path = '/DATA/gitanjali_2021cs03/prompthate/PromptHate-Code/save_model/modelall.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Make predictions on the test set
    predictions = predict(opt, model, test_loader)

    # Create a DataFrame with predictions and image IDs (or any other identifier)
    # df = pd.DataFrame(predictions, columns=['Probability_0', 'Probability_1'])
    # df['Image_ID'] = df['Image_ID'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)  # Evaluate only if it's a string

    # # Extract the 'label' field and create a new 'Prediction' column
    # df['Prediction'] = df['Image_ID'].apply(lambda x: 1 if x['label'] == 1 else 0)

    # # Save the DataFrame to a CSV file
    # df.to_csv('/DATA/gitanjali_2021cs03/prompthate/PromptHate-Code/predictions.csv', index=False)
    # # Now 'predictions' contains the predicted probabilities for each example in the test set.
    # # You can further process or save these predictions as needed.

    # Create a DataFrame with predictions
    df = pd.DataFrame(predictions, columns=['Probability_0', 'Probability_1'])

    # Save the DataFrame to a CSV file
    df.to_csv('/DATA/gitanjali_2021cs03/prompthate/PromptHate-Code/predictions_alliteration.csv', index=False)



    # # Create a DataFrame with a single column containing both Probability_0 and Probability_1
    # df = pd.DataFrame({'Probabilities': predictions.tolist()})

    # # Save the DataFrame to a CSV file
    # df.to_csv('/DATA/gitanjali_2021cs03/prompthate/PromptHate-Code/predictions.csv', index=False)