from preprocess.audio_process import *
from utils.test_loader import load_annotations,calculate_metrics,evaluate_model,plot_confusion_matrix
from nets.AVFDTI import AVFDTI
from config.test_config import CONFIG


def main():
    config = CONFIG
    val_lines = load_annotations(config["annotation_lines_val"])

    model = AVFDTI(kernel_num=config["kernel_num"],feature_dim=config["feature_dim"],num_class=config["num_class"])
    model = model.to(config["device"])
    model.eval()
    if config["checkpoint_path"]:
        model.load_state_dict(torch.load(config["checkpoint_path"]))

    gt_class, predict_class, real_positions, predicted_positions, class_right = evaluate_model(model, val_lines, config)
    calculate_metrics(gt_class, predict_class, real_positions, predicted_positions, config)
    # plot_confusion_matrix(gt_class, predict_class, config)

if __name__ == "__main__":
    main()
