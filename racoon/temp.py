import torch
import soundfile as sf

EFFB0_MODELS = None
RST101_MODELS = None
RXT50_MODELS = None

# Get predictions (logits)

prediction_dict = {}
with torch.no_grad():
    for audio_path in tqdm(all_audios):
        # with timer(f"Loading {str(audio_path)}", logger):
        clip, _ = sf.read(audio_path)

        seconds = []
        row_ids = []

        tot_len = int(np.ceil(len(clip) / 32000))
        if tot_len != 600:
            print("Warning: time is not 10 min", tot_len)

        for second in range(5, tot_len + 5, 5):
            row_id = "_".join(audio_path.name.split("_")[:2]) + f"_{second}"
            seconds.append(second)
            row_ids.append(row_id)

        test_df = pd.DataFrame({"row_id": row_ids, "seconds": seconds})
        dataset = TestDataset(df=test_df, clip=clip, chunk=[30, 20])
        loader = torchdata.DataLoader(dataset, batch_size=1, shuffle=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # prediction_dict = {}
        for data in loader:
            row_id = data["row_id"]
            prediction_dict[row_id] = {}

            image_5 = data["5sec_mel"].to(device)
            image_30 = data["30sec_mel"].to(device)
            image_20 = data["20sec_mel"].to(device)

            # ResNeSt 101
            rst101_prob_5 = None
            for model in RST101_MODELS:
                model.eval()
                pred_5 = model(image_5)
                prob_5 = pred_5["clipwise_output"].detach().cpu().numpy().reshape(-1)
                if rst101_prob_5 is None:
                    rst101_prob_5 += prob_5
            rst101_prob_5_avg = rst101_prob_5 / len(RST101_MODELS)
            prediction_dict[row_id]["rst101_prob_5_avg"] = rst101_prob_5_avg

            # Efficient-B0
            effb0_prob_5 = None
            effb0_prob_30 = None
            for model in EFFB0_MODELS:
                model.eval()
                pred_5 = model(image_5)
                prob_5 = pred_5["clipwise_output"].detach().cpu().numpy().reshape(-1)
                pred_30 = model(image_30)
                prob_30 = pred_30["clipwise_output"].detach().cpu().numpy().reshape(-1)
                if effb0_prob_5 is None:
                    effb0_prob_5 += prob_5
                    effb0_prob_30 += prob_30
            effb0_prob_5_avg = effb0_prob_5 / len(EFFB0_MODELS)
            effb0_prob_30_avg = effb0_prob_30 / len(EFFB0_MODELS)
            prediction_dict[row_id]["effb0_prob_5_avg"] = effb0_prob_5_avg
            prediction_dict[row_id]["effb0_prob_30_avg"] = effb0_prob_30_avg

            # ResNeXt50_32x4d
            rxt50_prob_5 = None
            rxt50_prob_20 = None
            for model in RXT50_MODELS:
                model.eval()
                pred_5 = model(image_5)
                prob_5 = pred_5["clipwise_output"].detach().cpu().numpy().reshape(-1)
                pred_20 = model(image_20)
                prob_20 = pred_20["clipwise_output"].detach().cpu().numpy().reshape(-1)
                if effb0_prob_5 is None:
                    rxt50_prob_5 += prob_5
                    rxt50_prob_20 += prob_20
            rxt50_prob_5_avg = rxt50_prob_5 / len(RXT50_MODELS)
            rxt50_prob_20_avg = rxt50_prob_20 / len(RXT50_MODELS)
            prediction_dict[row_id]["rxt50_prob_5_avg"] = rxt50_prob_5_avg
            prediction_dict[row_id]["rxt50_prob_20_avg"] = rxt50_prob_20_avg
