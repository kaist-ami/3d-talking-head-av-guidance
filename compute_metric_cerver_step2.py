import argparse
import json
import jiwer

class VisemeConverter:
    def __init__(self, phoneme_to_visemes_file):
        from phonemizer.backend import EspeakBackend
        from phonemizer.separator import Separator

        self.backend = EspeakBackend('en-us', words_mismatch='ignore', with_stress=False)
        self.separator = Separator(phone='-', word=' ')
        self.pho2vi, self.all_vis = self.get_phoneme_to_viseme_map(phoneme_to_visemes_file)

    def get_phoneme_to_viseme_map(self, phoneme_to_visemes_file):
        pho2vi = {}
        all_vis = []
        p2v = phoneme_to_visemes_file
        with open(p2v) as file:
            lines = file.readlines()
            for line in lines:
                if line.split(",")[0] in pho2vi:
                    if line.split(",")[4].strip() != pho2vi[line.split(",")[0]]:
                        print('error')
                pho2vi[line.split(",")[0]] = line.split(",")[4].strip()
                all_vis.append(line.split(",")[4].strip())
        return pho2vi, all_vis

    def convert_text_to_visemes(self, text):
        phonemized = self.backend.phonemize([text], separator=self.separator)[0]
        text = ""
        for word in phonemized.split(" "):
            visemized = []
            for phoneme in word.split("-"):
                if phoneme == "":
                    continue
                try:
                    visemized.append(self.pho2vi[phoneme.strip()])
                    if self.pho2vi[phoneme.strip()] not in self.all_vis:
                        self.all_vis.append(self.pho2vi[phoneme.strip()])
                except:
                    print('Count not find', phoneme)
                    continue
            text += " " + "".join(visemized)
        return text

def compute_cer_ver(data_list:list, viseme_converter:VisemeConverter, save_output_fpath, save_metric_fpath):
    """
    data_list (list): each element is {"key": ..., "pred_text": ..., "gt_text": ...}
    """
    cer_all = 0
    ver_all = 0
    results = []

    for item in data_list:
        key = item["key"]
        pred_text = item["pred_text"]
        gt_text = item["gt_text"]

        # CER
        cer = jiwer.cer(gt_text, pred_text)
        cer_all += cer

        # VER
        gt_viseme = viseme_converter.convert_text_to_visemes(gt_text)
        pred_viseme = viseme_converter.convert_text_to_visemes(pred_text)
        ver = jiwer.cer(gt_viseme, pred_viseme)
        ver_all += ver

        # Record results
        item = {
            "key": key,
            "pred_text": pred_text,
            "pred_viseme": pred_viseme,
            "gt_text": gt_text,
            "gt_viseme": gt_viseme,
            "CER": cer,
            "VER": ver,
        }
        results.append(item)

    cer_all /= len(data_list)
    ver_all /= len(data_list)
    print(f"CER: {cer_all}, VER: {ver_all}")

    with open(save_output_fpath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    with open(save_metric_fpath, "w") as f:
        f.write(f"# files: {len(data_list)}\n")
        f.write(f"CER: {cer_all}")
        f.write(f"VER: {ver_all}")


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Compute CER/VER"
    )
    parser.add_argument(
        "--pred_text_file", type=str, 
        default="outputs/vocaset/result/cer_ver.json",
        help="File that saving the GT and predicted text"
    )
    parser.add_argument(
        "--save_text_file", type=str,
        help="File to save CER and VER inference results per text"
    )
    parser.add_argument(
        "--save_metric_file", type=str, 
        help="File to save total CER, VER result"
    )
    parser.add_argument(
        "--phoneme_to_visemes_file", type=str, default="phonemes2visemes.csv",
        help="File that saving phoneme-to-viseme mapping"
    )
    cmd_input = parser.parse_args()

    if cmd_input.save_text_file is None:
        cmd_input.save_text_file = cmd_input.pred_text_file

    if cmd_input.save_metric_file is None:
        cmd_input.save_metric_file = (cmd_input.pred_text_file).replace(".json", ".txt")


    viseme_converter = VisemeConverter()
    with open(cmd_input.pred_text_file, "r", encoding="utf-8") as f:
        pred_text_list = json.load(f)
    compute_cer_ver(
        data_list=pred_text_list,
        viseme_converter=viseme_converter,
        save_output_fpath=cmd_input.save_text_file, 
        save_metric_fpath=cmd_input.save_metric_file
    )