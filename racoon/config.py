from pathlib import Path
import os

import argparse

# control here
parser = argparse.ArgumentParser()

parser.add_argument("--name", type=str, default="0528")
parser.add_argument("--use", type=int, default=2)


parser.add_argument("--period", type=int, default=30)
parser.add_argument("--n_mels", type=int, default=128)
parser.add_argument("--fmin", type=int, default=20)
parser.add_argument("--fmax", type=int, default=16000)
parser.add_argument("--n_fft", type=int, default=2048)
parser.add_argument("--hop_length", type=int, default=512)
parser.add_argument("--sample_rate", type=int, default=32000)

######################
# Globals #
######################
parser.add_argument("--gpu", "--g", type=str, default="3")
parser.add_argument("--seed", type=int, default=819)
parser.add_argument("--fold", "--f", type=int, default=0)
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--train", action="store_false")

parser.add_argument("--main_metric", type=str, default="epoch_f1_at_05")
parser.add_argument("--minimize_metric", action="store_true")


######################
# Data #
######################

parser.add_argument(
    "--train_datadir",
    default=Path("/data2/minki/kaggle/birdclef-2021/train_short_audio"),
)
parser.add_argument(
    "--train_csv",
    type=str,
    default="/data2/minki/kaggle/birdclef-2021/train_metadata.csv",
)
parser.add_argument(
    "--train_soundscape",
    type=str,
    default="/data2/minki/kaggle/birdclef-2021/train_soundscape_labels.csv",
)
parser.add_argument(
    "--background_datadir",
    type=str,
    default="/data2/minki/kaggle/birdclef-2021/background_soundscape",  # 32
)
parser.add_argument("--logdir", default=Path("/data2/minki/kaggle/birdclef-2021/ckpt"))

parser.add_argument("--mixup", action="store_true")
parser.add_argument("--use_secondary_label", "--sec", action="store_false")
parser.add_argument("--rating", type=float, default=2)

######################
# Loaders #
######################
parser.add_argument("--batch_size", "--b", type=int, default=32)
parser.add_argument("--num_workers", "--nw", type=int, default=8)


parser.add_argument(
    "--base_model_name", "--model", type=str, default="resnext50_32x4d"
)  # "tf_efficientnet_b0_ns", resnest101e, resnext50_32x4d, efficientnet_b0
parser.add_argument(
    "--loss_name", "--loss", type=str, default="BCE2WayLoss"
)  # "BCEFocal2WayLoss"
parser.add_argument("--optimizer_name", "--opt", type=str, default="Adam")
parser.add_argument("--base_optimizer", type=str, default="Adam")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument(
    "--scheduler_name",
    "--sched",
    type=str,
    default="CosineAnnealingLR",  # "CosineAnnealingWarmRestarts"
)

######################
# Custom #
######################
parser.add_argument("--resume_dir", "--resume", type=str, default=None)
parser.add_argument("--check", action="store_true")
parser.add_argument("--timeit", action="store_true")
parser.add_argument("--overfit", action="store_true")
parser.add_argument("--load_best_on_end", action="store_true")



######################
# Test #
######################
parser.add_argument(
    "--test_datadir", default=Path("/data2/minki/kaggle/birdclef-2021/test_soundscapes")
)

parser.add_argument("--test_weights_path", "--twp", type=Path)
parser.add_argument("--test_threshold", "--tthr", type=float, default=0.3)
parser.add_argument("--test_threshold_chunk", type=float, default=0.5)
parser.add_argument("--test_chunk", type=int, default=30)

CFG, _ = parser.parse_known_args()

CFG.melspectrogram_parameters = {
    "n_mels": CFG.n_mels,
    "fmin": CFG.fmin,
    "fmax": CFG.fmax,
}
CFG.loader_params = {
    "train": {
        "batch_size": CFG.batch_size,
        "num_workers": CFG.num_workers,
        "shuffle": True,
    },
    "valid": {
        "batch_size": CFG.batch_size * 2,
        "num_workers": CFG.num_workers,
        "shuffle": False,
    },
}
CFG.split = "StratifiedKFold"
CFG.split_params = {"n_splits": 5, "shuffle": True, "random_state": 52}
CFG.loss_params = {}
CFG.optimizer_params = {"lr": CFG.lr}
if CFG.scheduler_name == "CosineAnnealingLR":
    CFG.scheduler_params = {"T_max": 10}
elif CFG.scheduler_name == "CosineAnnealingWarmRestarts":
    CFG.scheduler_params = {"T_0": 10}

CFG.pooling = "max"
CFG.pretrained = True
CFG.num_classes = 397
CFG.in_channels = 1

CFG.transforms = {"train": [{"name": "Normalize"}], "valid": [{"name": "Normalize"}]}

os.environ["CUDA_VISIBLE_DEVICES"] = CFG.gpu

# CFG.logdir = f"/data2/minki/kaggle/ramdisk/out/{CFG.name}"
# os.makedirs(CFG.logdir, exist_ok=True)

CFG.target_columns = [
    "acafly",
    "acowoo",
    "aldfly",
    "ameavo",
    "amecro",
    "amegfi",
    "amekes",
    "amepip",
    "amered",
    "amerob",
    "amewig",
    "amtspa",
    "andsol1",
    "annhum",
    "astfly",
    "azaspi1",
    "babwar",
    "baleag",
    "balori",
    "banana",
    "banswa",
    "banwre1",
    "barant1",
    "barswa",
    "batpig1",
    "bawswa1",
    "bawwar",
    "baywre1",
    "bbwduc",
    "bcnher",
    "belkin1",
    "belvir",
    "bewwre",
    "bkbmag1",
    "bkbplo",
    "bkbwar",
    "bkcchi",
    "bkhgro",
    "bkmtou1",
    "bknsti",
    "blbgra1",
    "blbthr1",
    "blcjay1",
    "blctan1",
    "blhpar1",
    "blkpho",
    "blsspa1",
    "blugrb1",
    "blujay",
    "bncfly",
    "bnhcow",
    "bobfly1",
    "bongul",
    "botgra",
    "brbmot1",
    "brbsol1",
    "brcvir1",
    "brebla",
    "brncre",
    "brnjay",
    "brnthr",
    "brratt1",
    "brwhaw",
    "brwpar1",
    "btbwar",
    "btnwar",
    "btywar",
    "bucmot2",
    "buggna",
    "bugtan",
    "buhvir",
    "bulori",
    "burwar1",
    "bushti",
    "butsal1",
    "buwtea",
    "cacgoo1",
    "cacwre",
    "calqua",
    "caltow",
    "cangoo",
    "canwar",
    "carchi",
    "carwre",
    "casfin",
    "caskin",
    "caster1",
    "casvir",
    "categr",
    "ccbfin",
    "cedwax",
    "chbant1",
    "chbchi",
    "chbwre1",
    "chcant2",
    "chispa",
    "chswar",
    "cinfly2",
    "clanut",
    "clcrob",
    "cliswa",
    "cobtan1",
    "cocwoo1",
    "cogdov",
    "colcha1",
    "coltro1",
    "comgol",
    "comgra",
    "comloo",
    "commer",
    "compau",
    "compot1",
    "comrav",
    "comyel",
    "coohaw",
    "cotfly1",
    "cowscj1",
    "cregua1",
    "creoro1",
    "crfpar",
    "cubthr",
    "daejun",
    "dowwoo",
    "ducfly",
    "dusfly",
    "easblu",
    "easkin",
    "easmea",
    "easpho",
    "eastow",
    "eawpew",
    "eletro",
    "eucdov",
    "eursta",
    "fepowl",
    "fiespa",
    "flrtan1",
    "foxspa",
    "gadwal",
    "gamqua",
    "gartro1",
    "gbbgul",
    "gbwwre1",
    "gcrwar",
    "gilwoo",
    "gnttow",
    "gnwtea",
    "gocfly1",
    "gockin",
    "gocspa",
    "goftyr1",
    "gohque1",
    "goowoo1",
    "grasal1",
    "grbani",
    "grbher3",
    "grcfly",
    "greegr",
    "grekis",
    "grepew",
    "grethr1",
    "gretin1",
    "greyel",
    "grhcha1",
    "grhowl",
    "grnher",
    "grnjay",
    "grtgra",
    "grycat",
    "gryhaw2",
    "gwfgoo",
    "haiwoo",
    "heptan",
    "hergul",
    "herthr",
    "herwar",
    "higmot1",
    "hofwoo1",
    "houfin",
    "houspa",
    "houwre",
    "hutvir",
    "incdov",
    "indbun",
    "kebtou1",
    "killde",
    "labwoo",
    "larspa",
    "laufal1",
    "laugul",
    "lazbun",
    "leafly",
    "leasan",
    "lesgol",
    "lesgre1",
    "lesvio1",
    "linspa",
    "linwoo1",
    "littin1",
    "lobdow",
    "lobgna5",
    "logshr",
    "lotduc",
    "lotman1",
    "lucwar",
    "macwar",
    "magwar",
    "mallar3",
    "marwre",
    "mastro1",
    "meapar",
    "melbla1",
    "monoro1",
    "mouchi",
    "moudov",
    "mouela1",
    "mouqua",
    "mouwar",
    "mutswa",
    "naswar",
    "norcar",
    "norfli",
    "normoc",
    "norpar",
    "norsho",
    "norwat",
    "nrwswa",
    "nutwoo",
    "oaktit",
    "obnthr1",
    "ocbfly1",
    "oliwoo1",
    "olsfly",
    "orbeup1",
    "orbspa1",
    "orcpar",
    "orcwar",
    "orfpar",
    "osprey",
    "ovenbi1",
    "pabspi1",
    "paltan1",
    "palwar",
    "pasfly",
    "pavpig2",
    "phivir",
    "pibgre",
    "pilwoo",
    "pinsis",
    "pirfly1",
    "plawre1",
    "plaxen1",
    "plsvir",
    "plupig2",
    "prowar",
    "purfin",
    "purgal2",
    "putfru1",
    "pygnut",
    "rawwre1",
    "rcatan1",
    "rebnut",
    "rebsap",
    "rebwoo",
    "redcro",
    "reevir1",
    "rehbar1",
    "relpar",
    "reshaw",
    "rethaw",
    "rewbla",
    "ribgul",
    "rinkin1",
    "roahaw",
    "robgro",
    "rocpig",
    "rotbec",
    "royter1",
    "rthhum",
    "rtlhum",
    "ruboro1",
    "rubpep1",
    "rubrob",
    "rubwre1",
    "ruckin",
    "rucspa1",
    "rucwar",
    "rucwar1",
    "rudpig",
    "rudtur",
    "rufhum",
    "rugdov",
    "rumfly1",
    "runwre1",
    "rutjac1",
    "saffin",
    "sancra",
    "sander",
    "savspa",
    "saypho",
    "scamac1",
    "scatan",
    "scbwre1",
    "scptyr1",
    "scrtan1",
    "semplo",
    "shicow",
    "sibtan2",
    "sinwre1",
    "sltred",
    "smbani",
    "snogoo",
    "sobtyr1",
    "socfly1",
    "solsan",
    "sonspa",
    "soulap1",
    "sposan",
    "spotow",
    "spvear1",
    "squcuc1",
    "stbori",
    "stejay",
    "sthant1",
    "sthwoo1",
    "strcuc1",
    "strfly1",
    "strsal1",
    "stvhum2",
    "subfly",
    "sumtan",
    "swaspa",
    "swathr",
    "tenwar",
    "thbeup1",
    "thbkin",
    "thswar1",
    "towsol",
    "treswa",
    "trogna1",
    "trokin",
    "tromoc",
    "tropar",
    "tropew1",
    "tuftit",
    "tunswa",
    "veery",
    "verdin",
    "vigswa",
    "warvir",
    "wbwwre1",
    "webwoo1",
    "wegspa1",
    "wesant1",
    "wesblu",
    "weskin",
    "wesmea",
    "westan",
    "wewpew",
    "whbman1",
    "whbnut",
    "whcpar",
    "whcsee1",
    "whcspa",
    "whevir",
    "whfpar1",
    "whimbr",
    "whiwre1",
    "whtdov",
    "whtspa",
    "whwbec1",
    "whwdov",
    "wilfly",
    "willet1",
    "wilsni1",
    "wiltur",
    "wlswar",
    "wooduc",
    "woothr",
    "wrenti",
    "y00475",
    "yebcha",
    "yebela1",
    "yebfly",
    "yebori1",
    "yebsap",
    "yebsee1",
    "yefgra1",
    "yegvir",
    "yehbla",
    "yehcar1",
    "yelgro",
    "yelwar",
    "yeofly1",
    "yerwar",
    "yeteup1",
    "yetvir",
]


DEBUG = False
if DEBUG:
    CFG.epochs = 1
