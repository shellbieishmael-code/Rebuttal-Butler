"""Rebuttal Butler -- single-file desktop app.

How audio flows (all heavy work happens off the UI thread):

    sound card -> audio_q -> collector thread (resample + speech detection)
                                 |-> utter_q  (finished phrases, bounded)
                                 '-> partial slot (only the latest live snapshot)
                                                        |
                                              worker thread (Whisper)
                                                        |
                              events queue -> Tk main thread (polled every 50 ms)

Environment variables (all optional):
    MRAI_MODEL_SIZE   Whisper model, default "small.en". Smaller = faster:
                      base.en, tiny.en. Run benchmark.py to time them on your PC.
    MRAI_DEVICE       auto (default): use an NVIDIA GPU when one is usable, otherwise
                      the CPU. cuda: prefer the GPU. cpu: never use the GPU.
                      The GPU needs CUDA 12 cuBLAS and cuDNN 9; if they are missing the
                      app says so in the status line and uses the CPU.
    MRAI_COMPUTE      Number format: default int8 on cpu, float32 on cuda.
    MRAI_NO_GPU=1     Skip the GPU check entirely (same as MRAI_DEVICE=cpu).
    MRAI_CPU_THREADS  CPU threads for Whisper (default: about half your logical
                      cores, 4 to 8).
    MRAI_BEAM         Search width for finished phrases, default 2 (1 = fastest).
    MRAI_PARTIALS=0   Turn off the live "as they speak" text. Whisper then only
                      works on finished phrases, which helps slow CPUs.
    MRAI_END_SILENCE  Seconds of quiet that end a phrase, default 0.6.
    MRAI_SAVE_AUDIO=1 Save every detected phrase to ./debug_audio/*.wav so you can
                      listen to exactly what Whisper was given.
    MRAI_THEME        Window look: midnight (default), violet, daylight or stage.
                      You can also change DEFAULT_THEME in the "Look and feel" section.
    MRAI_FONT         Font for the window, default "Segoe UI".
    MRAI_DPI=0        Turn off the sharp-text mode for high-DPI screens (Windows).
    MRAI_MODELS_DIR   Folder holding Whisper models shipped with the app (default: the
                      "models" folder next to the app). A model found there is used
                      offline; otherwise it is downloaded on first run.
    MRAI_DATA_DIR     Where the library, log file and saved audio go (default: next to the
                      app, or %APPDATA%\\RebuttalButler if that folder is read-only).

Audio sources (pick one in the window):
    Desktop audio     Whatever this PC is playing, e.g. your browser dialer, captured with
                      Windows' own loopback. It is the same source as OBS's "Desktop Audio",
                      so neither OBS nor VB-Cable is needed. Your own microphone is not part
                      of it, so only the customer is transcribed.
    Input: ...        A microphone, headset or virtual cable, as before.

Run with --selftest to check the install and the audio capture (see run_selftest).
"""
import json
import math
import os
import queue
import re
import sys
import threading
import time
import traceback
import io
import wave
import zlib
from collections import deque
import tkinter as tk
from tkinter import ttk, messagebox, colorchooser, filedialog

import numpy as np

_DLL_FOLDERS = []                   # kept alive: Windows drops a folder when its handle is closed
_NVIDIA_FOLDERS = []               # the nvidia\\<name>\\bin folders that were found, for the self-test


def _add_nvidia_dll_folders():
    """Windows only looks in the app's top folder for DLLs. The NVIDIA libraries (cuBLAS,
    cuDNN) are packed in nvidia\\<name>\\bin subfolders, both in the packaged app and in a
    normal pip install, so those folders are added to the search path first."""
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return
    import glob
    import importlib.util
    roots = []
    if getattr(sys, "frozen", False):
        roots.append(os.path.join(getattr(sys, "_MEIPASS", ""), "nvidia"))
    else:
        try:
            spec = importlib.util.find_spec("nvidia")
            roots.extend(spec.submodule_search_locations or [] if spec else [])
        except Exception:
            pass
    for root in roots:
        for folder in sorted(glob.glob(os.path.join(root, "*", "bin"))):
            _NVIDIA_FOLDERS.append(folder)
            try:
                _DLL_FOLDERS.append(os.add_dll_directory(folder))
            except OSError:
                pass


def _nvidia_report():
    """Plain description of which NVIDIA files this app can see, for the self-test."""
    if not _NVIDIA_FOLDERS:
        return "the app contains no NVIDIA folders (nvidia\\...\\bin), so this build has no GPU files"
    parts = []
    for folder in _NVIDIA_FOLDERS:
        try:
            names = sorted(n for n in os.listdir(folder) if n.lower().endswith(".dll"))
        except OSError:
            names = []
        parts.append(f"{os.path.relpath(folder, getattr(sys, '_MEIPASS', APP_DIR))} ({len(names)} DLLs"
                     + (f": {', '.join(names[:3])}..." if names else "") + ")")
    return "NVIDIA folders found: " + "; ".join(parts)


_add_nvidia_dll_folders()

try:                                # microphones, headsets and virtual cables
    import sounddevice as sd
except Exception:                   # missing or no audio driver: desktop audio can still work
    sd = None
try:                                # Windows loopback = "desktop audio", no virtual cable needed
    import pyaudiowpatch as pyaudio
except Exception:                   # not Windows, or not installed
    pyaudio = None
from faster_whisper import WhisperModel

# ----------------------------------------------------------------------------
# Where files live (the same from a script and from the packaged .exe)
# ----------------------------------------------------------------------------
try:                                # plays back the saved MP3s on the Recordings tab
    import miniaudio
except ImportError:
    miniaudio = None
try:                                # MP3 recording (lameenc ships its own encoder)
    import lameenc
except ImportError:
    lameenc = None
try:                                # Windows' built-in sound player, for the click and recording sounds
    import winsound
except ImportError:
    winsound = None
FROZEN = bool(getattr(sys, "frozen", False))
APP_DIR = os.path.dirname(os.path.abspath(sys.executable if FROZEN else __file__))


def _is_writable(folder):
    try:
        os.makedirs(folder, exist_ok=True)
        probe = os.path.join(folder, f".write_test_{os.getpid()}")
        with open(probe, "w") as handle:
            handle.write("x")
        os.remove(probe)
        return True
    except OSError:
        return False


def _pick_data_dir():
    """The library, log and saved audio live next to the app when that folder can be
    written to (portable: unzip anywhere and run). Under Program Files it can't, so
    they go to %APPDATA% instead."""
    forced = os.environ.get("MRAI_DATA_DIR", "").strip()
    if forced and _is_writable(forced):
        return forced
    if not FROZEN or _is_writable(APP_DIR):
        return APP_DIR
    fallback = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "RebuttalButler")
    return fallback if _is_writable(fallback) else APP_DIR


DATA_DIR = _pick_data_dir()
MODEL_DIRS = [d for d in dict.fromkeys([
    os.environ.get("MRAI_MODELS_DIR", "").strip(),
    os.path.join(APP_DIR, "models"),
    os.path.join(DATA_DIR, "models"),
]) if d]


ASSET_DIR = os.path.join(getattr(sys, "_MEIPASS", None) or APP_DIR, "assets")
LOGO_FILE = os.path.join(ASSET_DIR, "rebuttal_butler_logo.png")
ICON_FILE = os.path.join(ASSET_DIR, "rebuttal_butler.ico")


def set_window_icon(window):
    """Windows only: the Rebuttal Butler icon in the title bar and taskbar."""
    if sys.platform == "win32" and os.path.isfile(ICON_FILE):
        try:
            window.iconbitmap(ICON_FILE)
        except tk.TclError:
            pass


def load_logo(target_height):
    """The logo as a Tk image exactly target_height pixels tall (or None when the file is
    missing). Resized smoothly with Pillow so the lines stay clean. If Pillow is not
    available, falls back to Tk's whole-step shrink, which looks rougher."""
    if not os.path.isfile(LOGO_FILE):
        return None
    target_height = max(1, int(target_height))
    try:
        from PIL import Image, ImageTk
        with Image.open(LOGO_FILE) as source:
            source = source.convert("RGBA")
            width = max(1, round(source.width * target_height / source.height))
            resized = source.resize((width, target_height), Image.LANCZOS)
        return ImageTk.PhotoImage(resized)
    except Exception:
        pass
    try:
        image = tk.PhotoImage(file=LOGO_FILE)
    except tk.TclError:
        return None
    step = max(1, -(-image.height() // target_height))
    return image.subsample(step, step) if step > 1 else image


def local_model_dir(name):
    """The folder of a model that ships with the app (models/<name>/model.bin), or None."""
    for base in MODEL_DIRS:
        folder = os.path.join(base, name)
        if os.path.isfile(os.path.join(folder, "model.bin")):
            return folder
    return None

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


SAMPLE_RATE = 16000                 # what Whisper expects
BLOCK_SECONDS = 0.10                # capture block size
# Use the model named in MRAI_MODEL_SIZE, else the best one that ships with the app, else small.en.
MODEL_SIZE = (os.environ.get("MRAI_MODEL_SIZE", "").strip()
              or next((n for n in ("small.en", "base.en", "tiny.en") if local_model_dir(n)), "small.en"))
SAVE_DEBUG_AUDIO = os.environ.get("MRAI_SAVE_AUDIO") == "1"

# Speed. Whisper always encodes a full 30-second window, so a 2-second phrase costs
# about the same as a 15-second one: speed comes from the model, the hardware and
# how many calls are made, not from phrase length.
def _gpu_status():
    """-> (usable, explanation). usable is True only when an NVIDIA GPU with CUDA is
    present AND its libraries load. AMD or Intel graphics, a missing driver or missing
    CUDA files all give False, so the app quietly uses the CPU instead."""
    if os.environ.get("MRAI_NO_GPU") == "1":
        return False, "GPU turned off by MRAI_NO_GPU=1"
    try:
        import ctranslate2
        devices = ctranslate2.get_cuda_device_count()
    except Exception as exc:
        return False, f"CUDA support could not start ({exc})"
    if devices < 1:
        return False, ("no NVIDIA GPU with a working driver was found (install or update the NVIDIA "
                       "driver if this PC has an NVIDIA card)")
    if sys.platform == "win32":
        import ctypes
        for dll in ("cublas64_12.dll", "cudnn64_9.dll"):
            try:
                ctypes.WinDLL(dll)
            except OSError as exc:
                return False, (f"NVIDIA GPU found, but {dll} could not be loaded ({exc}). "
                               f"{_nvidia_report()}. If the NVIDIA folders are missing or empty, rebuild "
                               "with build_windows.bat small.en gpu")
    return True, f"NVIDIA GPU found ({devices} device{'s' if devices != 1 else ''}) with CUDA libraries"


def _gpu_usable():
    return _gpu_status()[0]


# MRAI_DEVICE: "auto" (default) uses the GPU if there is a usable NVIDIA card and the
# CPU otherwise; "cuda" asks for the GPU (and still falls back to the CPU if it fails);
# "cpu" never touches the GPU.
_DEVICE_PREF = os.environ.get("MRAI_DEVICE", "auto").strip().lower()
if _DEVICE_PREF == "cuda":
    DEVICE = "cuda"
elif _DEVICE_PREF == "cpu":
    DEVICE = "cpu"
else:
    DEVICE = "cuda" if _gpu_usable() else "cpu"
# float32 is the safer GPU default: cards without tensor cores (such as the GTX 1650)
# often run float16 slower. Run benchmark.py --device cuda --compute float16 to compare.
COMPUTE_TYPE = os.environ.get("MRAI_COMPUTE", "").strip() or ("float32" if DEVICE == "cuda" else "int8")
CPU_THREADS = max(1, _env_int("MRAI_CPU_THREADS", min(8, max(4, (os.cpu_count() or 8) // 2))))
FINAL_BEAM = max(1, _env_int("MRAI_BEAM", 2))
PARTIALS_ENABLED = os.environ.get("MRAI_PARTIALS", "1").strip() != "0"

# Speech detection. Runs on every block and costs almost nothing, so Whisper only
# ever sees real phrases instead of silence/noise (which it turns into nonsense).
MIN_SPEECH_RMS = 0.004              # nothing quieter than this counts as speech
NOISE_MULTIPLIER = 3.0              # speech must be this much louder than the background
MAX_NOISE_FLOOR = 0.012             # cap so continuous talking can't raise the bar too far
NOISE_WINDOW_SECONDS = 8.0
END_SILENCE_SECONDS = _env_float("MRAI_END_SILENCE", 0.60)   # this much quiet ends a phrase
PREROLL_SECONDS = 0.30              # audio kept from just before speech starts
MIN_SPEECH_SECONDS = 0.40           # ignore clicks and pops shorter than this
MAX_UTTERANCE_SECONDS = 15.0        # force a cut if someone never pauses

# Live (replaceable) text while somebody is still talking.
PARTIAL_INTERVAL_SECONDS = _env_float("MRAI_PARTIAL_INTERVAL", 0.5 if DEVICE == "cuda" else 1.0)
PARTIAL_WINDOW_SECONDS = 8.0
MIN_PARTIAL_SECONDS = 1.0
PARTIAL_PAUSE_SECONDS = 0.30        # no live update once the speaker has paused this long

END_SILENCE_SAMPLES = int(END_SILENCE_SECONDS * SAMPLE_RATE)
MIN_SPEECH_SAMPLES = int(MIN_SPEECH_SECONDS * SAMPLE_RATE)
MAX_UTTERANCE_SAMPLES = int(MAX_UTTERANCE_SECONDS * SAMPLE_RATE)

# A natural sentence (not a keyword list) so Whisper can't "echo" it back as text.
USE_DOMAIN_PROMPT = True
DOMAIN_PROMPT = "A phone call between an insurance agent and a homeowner about mortgage protection."

OBJECTIONS = [
    {
        "title": "I already have life insurance",
        "category": "Existing coverage",
        "triggers": ["already have life insurance", "i have life insurance", "already have insurance", "covered already", 'i already have a life policy', 'i have a policy already', "i'm covered through work", 'my job gives me life insurance', 'i have life coverage', 'i have insurance through my employer', 'i already bought life insurance', "i'm already insured", 'my family is covered', "i don't need another life policy"],
        "response": (
            "That is good to hear. Most homeowners do have some form of coverage. The question isn't whether you have insurance. "
            "The question is whether it's structured to do what you want it to do if something happens. And that's why we review everything before making assumptions."
        ),
        "followup": "Is your current coverage mainly through work, or is it an individual policy?",
    },
    {
        "title": "I'm not interested",
        "category": "Rejection",
        "triggers": ["not interested", "no interest", "don't want it", "do not want it", 'no thanks', "i don't want to hear about it", "i'm not looking for insurance", 'not something i want', "i'll pass", 'i have no interest in this', 'please take me off your list', "i don't want coverage", "this isn't for me", "i'm not interested in discussing it"],
        "response": (
            "I understand. Before I let you go, can I ask what part you're not interested in—the coverage itself, "
            "or are you just not familiar with what the review involves? My job is to explain the options you're "
            "eligible to review so you can decide whether any of them make sense for your household."
        ),
        "followup": "What have you heard about mortgage protection so far?",
    },
    {
        "title": "Do I have to do this?",
        "category": "Requirement question",
        "triggers": ["have to do this", "is this required", "do i need to do this", "mandatory", "is it mandatory", 'am i required to do this', 'is this mandatory', 'do i have to get this', 'is this part of my mortgage', 'will my loan be affected', 'do i need to buy this', 'is this a requirement', 'can i say no', 'am i obligated to get coverage', 'is this compulsory'],
        "response": (
            "I'm calling to explain the protection options available for you to consider. You shouldn't assume "
            "you have to purchase a policy just because we're reviewing options. The purpose is to understand your "
            "situation, explain what may be available, and let you decide whether it's useful."
        ),
        "followup": "Was this mortgage from a purchase, a refinance, or a home-equity line?",
    },
    {
        "title": "What is this about?",
        "category": "Introduction",
        "triggers": ["what is this", "what's this about", "why are you calling", "what are you calling about", "explain this", 'why are you calling me', 'what are you talking about', 'what review', 'what does this concern', 'why did you contact me', 'what is this regarding', 'what are you offering', 'what is mortgage protection', 'what did you call about', "can you explain why you're calling"],
        "response": (
            "This is just the state regulated mortgage protection review. These benefits can include paying off the mortgage in case of death. "
            "Some can help make mortgage payments if you're disabled or can't work, and some programs may return premiums if the coverage is never used. "
            "Options and eligibility depend on the program, and some may involve underwriting, different rates, or a medical exam."
        ),
        "followup": "Was your home recently purchased, refinanced, or did you open a home-equity line?",
    },
    {
        "title": "Do I have to pay for this?",
        "category": "Cost",
        "triggers": ["have to pay", "does this cost", "is this free", "how much does it cost", "what does it cost", "pay for this", 'is there a charge', 'does this cost money', "what's the price", 'is it going to cost me', 'how much would i pay', 'is this included with my mortgage', 'am i paying for this', 'what is the monthly cost', 'is there a premium', 'do i have to buy anything'],
        "response": (
            "A policy would have a premium if you decide to apply for and accept coverage. There isn't one price for "
            "everyone; it depends on factors such as age, health, coverage amount, product, and underwriting. We can "
            "first see what options may be available and review the costs before you decide."
        ),
        "followup": "Would it help if we first looked at the coverage amount you would want to protect?",
    },
    {
        "title": "If you're selling something, I'm not buying",
        "category": "Sales resistance",
        "triggers": ["not buying", "if you're selling", "if you are selling", "trying to sell me", "sales pitch", "don't sell me", 'are you trying to sell me insurance', "i don't buy over the phone", "i'm not buying anything", 'this sounds like a sales call', "don't try to sell me something", "i know you're selling", "i'm not interested in a sales pitch", "i don't want to be sold to", "if this is a sales call i'm hanging up", "i don't purchase things on calls"],
        "response": (
            "I understand being cautious. I'm here to explain the options and their costs, not to assume you'll buy "
            "anything. If none of the options fit your needs or budget, you can decline. Let's establish what you're "
            "trying to protect first, then you can make an informed decision."
        ),
        "followup": "If you did consider coverage, would your main concern be the cost or what it covers?",
    },
    {
        "title": "I can't afford it",
        "category": "Affordability",
        "triggers": ["can't afford", "cannot afford", "too expensive", "on a fixed income", "no money", "tight budget", "i don't have room in my budget", "i can't add another bill", "i'm struggling financially", 'i live paycheck to paycheck', 'my budget is tight', "i can't afford another payment", "i'm on a limited income", 'i have too many bills', "i can't take on more expenses", 'that would be too expensive for me'],
        "response": (
            "I understand—monthly expenses matter, and I wouldn't want you to take on something that doesn't fit your "
            "budget. The first step is to see whether any suitable options exist and what they would actually cost. "
            "You can then decide whether the benefit is worth that expense, with no need to commit before reviewing the details."
        ),
        "followup": "Would you be comfortable reviewing an option only if the premium fit your budget?",
    },
    {
        "title": "I need to talk to my spouse",
        "category": "Decision-maker",
        "triggers": ["talk to my spouse", "talk to my wife", "talk to my husband", "ask my spouse", "discuss with my wife", "discuss with my husband", 'i need to ask my wife', 'i need to ask my husband', 'we make decisions together', "i won't decide without my spouse", "i want my partner's opinion", 'i need to run it by my partner', 'my spouse and i need to discuss it', "i can't decide on my own", 'let me talk with my wife first', 'let me talk with my husband first'],
        "response": (
            "That makes sense—it's a household decision. I can help you understand the options and put the key details "
            "together so you can discuss them. We don't need to make a decision before both of you are comfortable with the information."
        ),
        "followup": "What do you think your spouse will want to know first: cost, coverage, or how it works?",
    },
    {
        "title": "My spouse isn't here",
        "category": "Decision-maker",
        "triggers": ["spouse isn't here", "spouse is not here", "wife isn't here", "husband isn't here", "they're not here", "they are not here", "my wife isn't home", "my husband isn't home", 'my partner is at work', 'i need my spouse on the phone', "they can't talk right now", 'my wife is unavailable', 'my husband is unavailable', 'we should both be on this call', "i can't speak for both of us", "call when we're both home"],
        "response": (
            "No problem. We can keep this informational and identify the questions you both would want answered. If you "
            "prefer, we can arrange a time when you can both listen and review the details together."
        ),
        "followup": "Would a time when you're both available work better?",
    },
    {
        "title": "How did you get my information?",
        "category": "Privacy / source",
        "triggers": ["how did you get my information", "where did you get my number", "how did you get my number", "who gave you my information", 'where did you get my details', 'how did you get my contact info', 'who gave you my phone number', 'why do you have my number', 'where did my information come from', 'how did you find me', 'who provided my details', 'how did you know about my mortgage', 'where did you get my address', 'why am i on your list'],
        "response": (
            "That's a fair question. I don't want to guess about the source of your information. I can explain who I "
            "represent and what information is being used for this contact, and if you'd like, we can verify the details before continuing."
        ),
        "followup": "Would you like me to identify the organization I'm calling on behalf of first?",
    },
    {
        "title": "If I die, they'll just sell the house",
        "category": "Perceived need",
        "triggers": ["sell the house", "they can sell the house", "just sell my house", "family will sell", "they'll sell it", 'my family can sell the home', 'they can just put the house on the market', "they won't keep the house", 'the house will be sold anyway', 'my kids can sell it', 'they can use the equity', "they'll move somewhere else", "i don't care if they sell it", 'selling the house would solve it', "they won't need to keep my house"],
        "response": (
            "That may be what your family chooses, and it's entirely their decision. The question is whether you want "
            "them to have other options and time to decide, rather than feeling pressured by mortgage payments or needing "
            "to make a quick decision. Coverage costs money, so it comes down to whether that flexibility is worth considering for you."
        ),
        "followup": "Would your priority be helping them keep the home, or giving them financial flexibility?",
    },
    {
        "title": "Why wasn't this offered before?",
        "category": "Prior process",
        "triggers": ["why wasn't this offered", "why was this not offered", "nobody offered this", "never told me about this", "why didn't anyone tell me earlier", "why wasn't i told at closing", 'why am i hearing about this now', "why didn't my lender mention it", "why wasn't this included in the loan", "why didn't the bank offer this", 'i never heard of this before', "why wasn't this brought up when i refinanced", "why didn't anyone explain this before", "shouldn't this have been offered at closing"],
        "response": (
            "I can't speak to what was discussed previously without seeing those details. Mortgage protection is separate "
            "from your mortgage itself, and availability and eligibility depend on the product and circumstances. I'm here "
            "to explain the options now so you can decide whether you'd like to consider them."
        ),
        "followup": "Was the mortgage recently purchased, refinanced, or is this related to a home-equity line?",
    },
    {
        "title": "My bank said I don't need this",
        "category": "Bank / lender",
        "triggers": ["bank said i don't need", "bank said i do not need", "lender said i don't need", "my bank said", "the lender said this wasn't necessary", 'my bank never said i needed it', "the bank told me i'm fine", 'my loan officer said no', "my lender said i'm covered", 'the bank said it was optional', "my mortgage company didn't require it", 'i checked with my bank and they said no', "the lender says i'm protected", "my bank says i don't need extra coverage"],
        "response": (
            "Your bank can explain its own loan requirements, and you should follow up with them if you have questions "
            "about those. This review is separate: it looks at whether you want personal coverage to help your household "
            "if a covered event happens. It's optional unless a specific agreement says otherwise, and the policy details matter."
        ),
        "followup": "Were you told about homeowners insurance or mortgage insurance, or about a separate life policy?",
    },
    {
        "title": "I already have homeowners insurance or PMI",
        "category": "Other insurance",
        "triggers": ["homeowners insurance", "homeowners policy", "pmi", "mortgage insurance", "insurance through the bank", 'i have home insurance', 'my homeowners policy covers the house', 'i already pay pmi', 'i have mortgage insurance', 'my lender has insurance on the property', 'my home is already insured', 'i have insurance with my mortgage', 'my house has coverage', "isn't pmi the same thing", 'my homeowners insurance should cover that'],
        "response": (
            "Those products generally serve different purposes. Homeowners insurance typically covers specified damage "
            "to the property, while PMI generally protects the lender against certain losses if a loan defaults. Personal "
            "life insurance is different and may provide a benefit to named beneficiaries, depending on the policy. We can "
            "compare the purposes without assuming you need another policy."
        ),
        "followup": "Do you mean homeowners insurance, PMI, or an individual life insurance policy?",
    },
    {
        "title": "A realtor or mortgage broker handled my loan",
        "category": "Third party",
        "triggers": ["realtor", "real estate agent", "mortgage broker", "loan officer handled", "my broker", 'my realtor took care of everything', 'my real estate agent handled it', 'my mortgage broker covered that', 'my loan officer went over everything', 'the broker already explained this', 'my realtor never mentioned this', 'the agent handled the paperwork', 'my lender handled all the details', 'i went through a mortgage broker', 'my real estate person took care of it'],
        "response": (
            "They helped with the home transaction or financing, which is a different role from reviewing a separate "
            "personal insurance policy. I'm not suggesting they did anything wrong. I'm offering to explain a different "
            "type of protection so you can decide whether it's relevant to your circumstances."
        ),
        "followup": "Did they discuss any personal life coverage with you, or mainly the loan and closing?",
    },
    {
        "title": "I'm busy / call me later",
        "category": "Scheduling",
        "triggers": ["i'm busy", "im busy", "call me later", "not a good time", "can't talk now", "cannot talk now", "i'm at work", "i can't talk right now", "i'm in the middle of something", 'can you call another time', "i don't have time right now", "i'm driving", 'this is a bad time', 'call me back later today', "i'm occupied right now", 'can we do this later'],
        "response": (
            "Of course. I don't want to interrupt your day. Let's find a better time when you can give this your attention, "
            "and I'll keep the review focused."
        ),
        "followup": "What day and time would be more convenient?",
    },
    {
        "title": "How does the death benefit work?",
        "category": "Policy mechanics",
        "triggers": ["cash benefit", "death benefit", "beneficiary", "payout", "how does it pay", "face amount", 'who gets paid when i die', 'how much would my family receive', 'what does the policy pay out', 'what is the face value', 'does the beneficiary get the money', 'how is the payout calculated', 'does it pay the mortgage directly', 'what happens to the benefit', 'how does the payout work', 'would my family get the full amount'],
        "response": (
            "The face amount is the policy's stated death benefit, subject to the contract's terms, and the benefit is "
            "generally paid to the named beneficiary or beneficiaries if the insured dies while coverage is in force. "
            "Loans, outstanding interest, exclusions, and other policy provisions may affect the amount paid. The exact details depend on the policy."
        ),
        "followup": "Would you like to focus on who receives the benefit or how much coverage might be appropriate?",
    },
    {
        "title": "I need to think about it",
        "category": "Delay",
        "triggers": ["think about it", "need to think", "let me think", "not ready to decide", "give me some time", 'i need more time', "i'm not ready to decide", 'let me sleep on it', 'i want to think this over', "i'll get back to you", 'i need to consider my options', "i don't want to decide today", 'can i think about it first', 'i need time to review it', 'i want to look it over'],
        "response": (
            "Absolutely. It's worth understanding what you're considering before making a decision. Let's make sure you "
            "have the coverage details, premium, limitations, and any exclusions so you can weigh them properly. Reviewing "
            "information doesn't obligate you to purchase."
        ),
        "followup": "What part would you most want clarified before you think it over?",
    },
]


# ----------------------------------------------------------------------------
# Text helpers and objection matching
# ----------------------------------------------------------------------------
LIBRARY_FILE = os.path.join(DATA_DIR, "rebuttal_library.json")
SCRIPT_FILE = os.path.join(DATA_DIR, "script_sections.json")
DEFAULT_SCRIPT_SECTIONS = ["SCRIPT", "CRITICAL PERIOD CLOSE", "MORTGAGE PROTECTION", "PHONE SCRIPT (A,AA)",
                           "QUOTING GUIDELINES", "D LEAD PHONE", "A1 PHONE SCRIPT"]


def load_script_sections():
    """The call script sections as [{"name": ..., "text": ...}]. Blank until filled in."""
    try:
        with open(SCRIPT_FILE, "r", encoding="utf-8") as file:
            raw = json.load(file)
        if isinstance(raw, list):
            return [{"name": str(item.get("name", "")).strip() or "Untitled", "text": str(item.get("text", ""))}
                    for item in raw if isinstance(item, dict)]
    except (OSError, ValueError, TypeError):
        pass
    return [{"name": name, "text": ""} for name in DEFAULT_SCRIPT_SECTIONS]


def save_script_sections(sections):
    with open(SCRIPT_FILE, "w", encoding="utf-8") as file:
        json.dump(sections, file, indent=2, ensure_ascii=False)
USER_ADDED_OBJECTIONS = []

# Transcript highlighter: the notes offered in the drop-down (you can also type your own).
NOTE_CHOICES = ["Needs rebuttal", "Good rebuttal", "Rebuttal didn't work", "Customer objection",
                "Follow up", "Important", "Sale / next step", "Wrong match"]
HIGHLIGHT_COLORS = [("Yellow", "#F6D365"), ("Green", "#7BE0A3"), ("Red", "#F28B82"),
                    ("Blue", "#8AB4F8"), ("Purple", "#C9A7F5")]
TRANSCRIPT_FOLDER_NAME = "Saved transcripts"
AUDIO_FOLDER_NAME = "audiofiles"
RECORD_CHOICES = [("My side", "mine"), ("Caller", "caller"), ("Both", "both")]     # the Record dropdown
TUTORIAL_FILE = os.path.join(DATA_DIR, "tutorial_done.txt")   # created once the tour is finished or skipped

# First-run tour: one window per step, each closed with its OK button. (tab to show, title, text)
TUTORIAL_STEPS = [
    ("Live Assistant", None, "Welcome to Rebuttal Butler",
     "This short tour shows you each part of the app, one step at a time. Click OK to go to the next step. "
     "You can skip the tour at any time, and replay it later with the Replay Tutorial button."),
    ("Settings", "audio", "Choose what to listen to",
     "On the Settings tab, pick Desktop audio to hear your browser dialer through this PC's speakers or headset. "
     "Pick your own microphone too. Use Refresh Devices if a source is missing, and Test Audio to check the sound."),
    ("Live Assistant", "start", "Start and stop listening",
     "Click Start Listening when the call begins. The status light at the top right turns green while the app "
     "is listening. Click Stop Listening when the call ends."),
    ("Live Assistant", "preview", "See what was heard",
     "This box shows the last thing the customer said. The Transcript below it keeps every phrase with its time."),
    ("Live Assistant", "popup", "The Suggested Rebuttal window",
     "When an objection is recognized, this window opens. Read the SAY THIS script, and use the ASK NEXT question "
     "to keep the conversation going. Good Match marks the match in the transcript, Other Suggestion shows the next "
     "match, Copy Response copies the script, and Hide closes the window."),
    ("Live Assistant", "highlight", "Highlight important lines",
     "Select text in the Transcript, pick a color and a note from the two boxes beside Highlight (for example "
     "Needs rebuttal), then click Highlight Selection. Remove Highlight clears the highlight from the selected "
     "text. You can type your own note into the note box if none of the choices fit."),
    ("Live Assistant", "save", "Save the transcript",
     "Click Save Transcript to save the transcript and all of your highlights as a text file. Files go in the "
     "Saved transcripts folder, inside the Rebuttal Butler folder next to RebuttalButler.exe."),
    ("Live Assistant", "manual", "Find a rebuttal by hand",
     "Type a few words from the objection into Find a rebuttal manually, then click Search. The suggestions "
     "appear in the Suggested Rebuttal window."),
    ("Add Rebuttal", "tab:Add Rebuttal", "Add your own rebuttals",
     "Enter the objection, the other ways a customer might say it, and your approved script. Then click "
     "Accept & Add to Library to save it."),
    ("Library", "tab:Library", "Edit the library",
     "Pick a rebuttal in the list to view or change it. Then use Save Changes to keep your edits, Add New to "
     "start a new rebuttal, or Delete Selected to remove one."),
    ("Settings", "tab:Settings", "Make it yours",
     "On the Settings tab you can choose a theme, change any color, set the text size and font, and set up the "
     "Suggested Rebuttal window. Changes apply straight away."),
    ("Live Assistant", None, "You're all set",
     "That is the whole app. You can replay this tour at any time with the Replay Tutorial button on the Live "
     "Assistant tab."),
]


def normalize(text):
    text = (text or "").lower().replace("’", "'").replace("‘", "'")
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_library():
    """Load saved edits and user entries, matching built-ins by title."""
    try:
        with open(LIBRARY_FILE, "r", encoding="utf-8") as file:
            saved = json.load(file)
        if not isinstance(saved, list):
            return
        valid_saved = []
        for saved_item in saved:
            if not isinstance(saved_item, dict):
                continue
            title = str(saved_item.get("title", "")).strip()
            response = str(saved_item.get("response", "")).strip()
            if not title or not response:
                continue
            saved_item.setdefault("category", "User added")
            saved_item.setdefault("triggers", [title])
            saved_item.setdefault("followup", "")
            valid_saved.append(saved_item)

        built_in_titles = {normalize(item.get("title", "")) for item in OBJECTIONS}
        saved_titles = {normalize(item.get("title", "")) for item in valid_saved}
        if built_in_titles & saved_titles:
            # A file containing built-in titles is a full saved snapshot, so deletions persist too.
            OBJECTIONS[:] = valid_saved
        else:
            # Backward compatibility: older versions saved only user-created entries.
            existing_titles = {normalize(item.get("title", "")) for item in OBJECTIONS}
            for item in valid_saved:
                key = normalize(item.get("title", ""))
                if key not in existing_titles:
                    OBJECTIONS.append(item)
                    existing_titles.add(key)
        USER_ADDED_OBJECTIONS[:] = [item for item in OBJECTIONS if normalize(item.get("category", "")) == "user added"]
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"Could not load saved rebuttal library: {exc}")


def save_user_library():
    """Persist the complete editable library, including changes to built-ins."""
    try:
        with open(LIBRARY_FILE, "w", encoding="utf-8") as file:
            json.dump(OBJECTIONS, file, indent=2, ensure_ascii=False)
    except Exception as exc:
        raise OSError(f"Could not save library file: {exc}") from exc


# Everyday words that say nothing about WHICH objection a customer is raising.
# The old matcher counted these, so "do I have to go to the store" matched
# "do I have to get this". They are ignored when judging fuzzy matches.
STOPWORDS = frozenset(
    "a an the and or but so to of in on at by for with from as i me my we us our you your he she it its they them "
    "their is am are was were be do does did this that these those um uh oh yeah well just".split()
)


def _content_words(tokens):
    return [t for t in tokens if t not in STOPWORDS]


def _ordered_match(trigger_tokens, utter_tokens):
    """Fuzzy match that survives Whisper dropping or changing a word or two.

    The trigger's meaningful words must appear in the same order and close together.
    Returns how many matched (0 means no match)."""
    words = _content_words(trigger_tokens)
    if len(words) < 2:
        return 0
    need = max(2, math.ceil(0.75 * len(words)))
    max_span = len(trigger_tokens) + 1
    best = 0
    for start in range(len(utter_tokens)):
        pos, first, matched = start, None, 0
        for word in words:
            try:
                idx = utter_tokens.index(word, pos)
            except ValueError:
                continue
            if first is None:
                first = idx
            elif idx - first + 1 > max_span:
                continue
            matched += 1
            pos = idx + 1
        if matched >= need:
            best = max(best, matched)
    return best


def match_objections(text, limit=4, strict=False):
    """Return the best matching objections for `text`.

    strict=True only accepts exact trigger phrases (used on unfinished, jittery
    live text). Otherwise a close in-order match on the meaningful words also counts."""
    normalized = normalize(text)
    if not normalized:
        return []
    padded = f" {normalized} "
    tokens = normalized.split()
    scored = []
    for item in OBJECTIONS:
        score = 0.0
        for trigger in item.get("triggers", []):
            phrase = normalize(trigger)
            if not phrase:
                continue
            if f" {phrase} " in padded:
                score = max(score, 10.0 + len(phrase.split()))
            elif not strict:
                matched = _ordered_match(phrase.split(), tokens)
                if matched:
                    score = max(score, 3.0 + 0.5 * matched)
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:limit]]


load_library()


# ----------------------------------------------------------------------------
# Audio helpers
# ----------------------------------------------------------------------------
def to_16k(mono, rate):
    """Convert a mono float32 block recorded at `rate` Hz to 16 kHz."""
    if rate == SAMPLE_RATE or len(mono) == 0:
        return mono
    if rate % SAMPLE_RATE == 0:
        k = rate // SAMPLE_RATE
        usable = (len(mono) // k) * k
        return mono[:usable].reshape(-1, k).mean(axis=1).astype(np.float32)
    smoothed = np.convolve(mono, np.ones(3, dtype=np.float32) / 3.0, mode="same")
    out_len = max(1, int(round(len(mono) * SAMPLE_RATE / rate)))
    positions = np.linspace(0, len(mono) - 1, out_len)
    return np.interp(positions, np.arange(len(mono)), smoothed).astype(np.float32)


def prepare_audio(audio):
    """Bring quiet phrases up to a healthy level before recognition."""
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if 0.003 < peak < 0.5:
        audio = (audio * min(0.7 / peak, 20.0)).astype(np.float32)
    return audio


HALLUCINATION_PHRASES = (
    "thanks for watching", "thank you for watching", "subscribe", "amara org", "subtitles by", "captions by",
)


def looks_like_hallucination(text):
    """Catch the stock phrases Whisper invents for silence and noise."""
    cleaned = normalize(text)
    if not cleaned:
        return True
    if any(phrase in cleaned for phrase in HALLUCINATION_PHRASES):
        return True
    words = cleaned.split()
    if len(words) >= 6 and len(set(words)) <= 2:          # "you you you you you you"
        return True
    prompt = normalize(DOMAIN_PROMPT)
    if len(words) >= 4 and cleaned in prompt:               # model parroting its own prompt
        return True
    return False


class UtteranceSegmenter:
    """Cuts a continuous audio stream into phrases using loudness.

    Feed it 16 kHz mono blocks. It reports each finished phrase, and can give a
    snapshot of the phrase still in progress."""

    def __init__(self):
        self.recent_rms = deque(maxlen=max(20, int(NOISE_WINDOW_SECONDS / BLOCK_SECONDS)))
        self.preroll = deque()
        self.preroll_samples = 0
        self.next_id = 1
        self._reset()

    def _reset(self):
        self.in_speech = False
        self.utt_id = 0
        self.chunks = []
        self.total_samples = 0
        self.speech_samples = 0
        self.silence_samples = 0

    def _threshold(self):
        # The background level is the quiet end of the last few seconds. Needs a
        # little history first, so the very first words of a call are never missed.
        if len(self.recent_rms) < 20:
            return MIN_SPEECH_RMS
        floor = min(float(np.percentile(self.recent_rms, 10)), MAX_NOISE_FLOOR)
        return max(MIN_SPEECH_RMS, floor * NOISE_MULTIPLIER)

    def feed(self, block):
        """Returns (rms, finished) where finished is a list of (utt_id, audio)."""
        n = len(block)
        rms = float(np.sqrt(np.mean(np.square(block)))) if n else 0.0
        loud = rms >= self._threshold()
        self.recent_rms.append(rms)
        finished = []

        if not self.in_speech:
            if loud:
                self.in_speech = True
                self.utt_id = self.next_id
                self.next_id += 1
                self.chunks = list(self.preroll) + [block]
                self.total_samples = self.preroll_samples + n
                self.speech_samples = n
                self.silence_samples = 0
                self.preroll.clear()
                self.preroll_samples = 0
            else:
                self.preroll.append(block)
                self.preroll_samples += n
                limit = int(PREROLL_SECONDS * SAMPLE_RATE)
                while self.preroll and self.preroll_samples - len(self.preroll[0]) >= limit:
                    self.preroll_samples -= len(self.preroll.popleft())
            return rms, finished

        self.chunks.append(block)
        self.total_samples += n
        if loud:
            self.speech_samples += n
            self.silence_samples = 0
        else:
            self.silence_samples += n
        if self.silence_samples >= END_SILENCE_SAMPLES or self.total_samples >= MAX_UTTERANCE_SAMPLES:
            if self.speech_samples >= MIN_SPEECH_SAMPLES:
                finished.append((self.utt_id, np.concatenate(self.chunks)))
            self._reset()
        return rms, finished

    def snapshot(self):
        """The phrase in progress as (utt_id, audio), or None if there isn't enough yet."""
        if not self.in_speech:
            return None
        # The speaker has paused, so the finished phrase is about to be queued. A live
        # update started now could not be interrupted and would only delay it.
        if self.silence_samples >= PARTIAL_PAUSE_SECONDS * SAMPLE_RATE:
            return None
        audio = np.concatenate(self.chunks)
        if len(audio) < MIN_PARTIAL_SECONDS * SAMPLE_RATE:
            return None
        return self.utt_id, audio[-int(PARTIAL_WINDOW_SECONDS * SAMPLE_RATE):]


# ----------------------------------------------------------------------------
# Live transcription engine (no UI code in here)
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Desktop audio: what this PC is playing (Windows loopback)
# ----------------------------------------------------------------------------
def _clean_device_name(name):
    return re.sub(r"\s*\[Loopback\]\s*$", "", str(name or "")).strip() or "Speakers"


def desktop_audio_info():
    """-> (name of the default output, [(device_index, name), ...]) for every output the
    PC can be heard on. Empty when this isn't Windows or nothing can be listed."""
    if pyaudio is None:
        return "", []
    pa = None
    try:
        pa = pyaudio.PyAudio()
        try:
            default_name = _clean_device_name(pa.get_default_wasapi_loopback()["name"])
        except Exception:
            default_name = ""
        outputs = [(int(d["index"]), _clean_device_name(d["name"]))
                   for d in pa.get_loopback_device_info_generator()]
        return default_name, outputs
    except Exception:
        return "", []
    finally:
        if pa is not None:
            try:
                pa.terminate()
            except Exception:
                pass


class DesktopAudioStream:
    """Captures what this PC is playing through Windows' own loopback (WASAPI), the same
    source OBS calls "Desktop Audio", so no virtual cable is needed.

    It looks like a sounddevice stream (start / stop / close) and hands `callback` float32
    blocks of BLOCK_SECONDS shaped (frames, channels). `device_index` None means "whatever
    the default speakers are right now"."""

    def __init__(self, device_index, callback):
        if pyaudio is None:
            raise RuntimeError("Desktop audio needs Windows (the PyAudioWPatch library is missing). "
                               "Pick a microphone or virtual cable from the list instead.")
        self.sink = callback
        self.pa = None
        self.stream = None
        self.name = ""
        self.rate = 48000
        self.channels = 2
        self._fmt = None
        self._pending = []
        self._pending_frames = 0
        try:
            self.pa = pyaudio.PyAudio()
            if device_index is None:
                try:
                    info = self.pa.get_default_wasapi_loopback()
                except (OSError, LookupError) as exc:
                    raise RuntimeError("Windows reports no speakers or headset to listen to. Plug one in "
                                       "(or switch the output on), then click Refresh Devices.") from exc
            else:
                try:
                    info = self.pa.get_device_info_by_index(int(device_index))
                except Exception as exc:
                    raise RuntimeError("That output is no longer available. Click Refresh Devices "
                                       "and choose again.") from exc
            self.name = _clean_device_name(info.get("name"))
            self.rate = int(info.get("defaultSampleRate") or 48000)
            native = max(1, int(info.get("maxInputChannels") or 2))
            self.block_frames = max(1, int(self.rate * BLOCK_SECONDS))
            formats = [(pyaudio.paFloat32, native), (pyaudio.paInt16, native)]
            if native != 2:
                formats += [(pyaudio.paFloat32, 2), (pyaudio.paInt16, 2)]
            # Small driver chunks first (steadier on Windows); 0 lets the driver choose.
            attempts = [(fmt, ch, chunk) for fmt, ch in formats for chunk in (1024, 0)]
            last_error = None
            for fmt, ch, chunk in attempts:
                try:
                    self.stream = self.pa.open(
                        format=fmt, channels=ch, rate=self.rate, input=True,
                        input_device_index=int(info["index"]), frames_per_buffer=chunk,
                        start=False, stream_callback=self._on_data)
                    self._fmt, self.channels = fmt, ch
                    break
                except Exception as exc:
                    last_error = exc
            if self.stream is None:
                raise RuntimeError(f"Could not open desktop audio on '{self.name}': {last_error}")
        except Exception:
            self.close()
            raise

    def start(self):
        self.stream.start_stream()

    def stop(self):
        stream = self.stream
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass

    def close(self):
        stream, self.stream = self.stream, None
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        pa, self.pa = self.pa, None
        if pa is not None:
            try:
                pa.terminate()
            except Exception:
                pass

    def _on_data(self, in_data, frame_count, time_info, status):
        # Runs on the audio driver's thread: keep it tiny. Windows may hand over very small
        # or odd-sized chunks, so they are regrouped into steady BLOCK_SECONDS blocks here.
        try:
            if in_data:
                if self._fmt == pyaudio.paInt16:
                    data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
                else:
                    data = np.frombuffer(in_data, dtype=np.float32)
                data = data[: len(data) // self.channels * self.channels].reshape(-1, self.channels)
                self._pending.append(data)
                self._pending_frames += len(data)
                while self._pending_frames >= self.block_frames:
                    joined = np.concatenate(self._pending) if len(self._pending) > 1 else self._pending[0]
                    block, rest = joined[: self.block_frames], joined[self.block_frames:]
                    self._pending = [rest] if len(rest) else []
                    self._pending_frames = len(rest)
                    self.sink(np.array(block, dtype=np.float32), self.block_frames, time_info, status)
        except Exception:
            pass
        return (None, pyaudio.paContinue)


class CallRecorder:
    """Writes ONE MP3 for a whole recording. Both sides (the caller and your microphone) are mixed
    into one signal, and that signal is written to every output channel: a stereo file has the
    full mix in left AND right, a mono file has it in its one channel. Each side is buffered as it
    is captured and written in step with the other, so they stay together to about a tenth of a
    second. Written as it goes, so nothing is lost if the app closes early."""

    RATE = 16000

    def __init__(self, path, channels):
        if lameenc is None:
            raise RuntimeError("MP3 recording needs the lameenc library, which is missing.")
        self.path = path
        self.channels = channels
        self._lock = threading.RLock()
        self._buffers = [np.zeros(0, dtype=np.float32) for _ in range(2)]     # 0 = caller, 1 = mic
        self._file = open(path, "wb")
        self._encoder = lameenc.Encoder()
        self._encoder.set_bit_rate(96)
        self._encoder.set_in_sample_rate(self.RATE)
        self._encoder.set_channels(channels)
        self._encoder.set_quality(2)
        self._open = True
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()

    def feed(self, channel, samples):
        if channel >= len(self._buffers) or len(samples) == 0:
            return
        with self._lock:
            if not self._open:
                return
            joined = np.concatenate((self._buffers[channel], np.asarray(samples, dtype=np.float32)))
            limit = self.RATE * 2                   # never more than two seconds behind: drop the oldest
            self._buffers[channel] = joined[-limit:] if len(joined) > limit else joined

    def _writer(self):
        while not self._stop.wait(0.1):
            self._write_ready()

    def _write_ready(self):
        with self._lock:
            if not self._open:
                return
            count = max(len(buffer) for buffer in self._buffers)
            if count == 0:
                return
            takes = []
            for index, buffer in enumerate(self._buffers):
                take = buffer[:count]
                self._buffers[index] = buffer[count:]
                takes.append(np.pad(take, (0, count - len(take))))     # a quiet side gets silence
            mixed = np.clip(takes[0] + takes[1], -1.0, 1.0)
            data = np.repeat(mixed.reshape(-1, 1), self.channels, axis=1)  # same mix in every channel
            pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            self._file.write(self._encoder.encode(pcm))

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        with self._lock:
            if not self._open:
                return
            self._write_ready()
            self._open = False
            try:
                self._file.write(self._encoder.flush())
            finally:
                self._file.close()


SOUND_RATE = 44100
_SOUND_LEVEL = {"hover": 0.15, "select": 0.22, "record_start": 0.9, "record_stop": 0.9}
_SOUND_CACHE = {}
_hover_clock = [0.0]


def _sound_wave(name):
    """A sound as float samples, made here so the app ships no sound files."""
    rate = SOUND_RATE

    def tone(freq, seconds, decay):
        t = np.arange(int(seconds * rate)) / rate
        return np.sin(2 * np.pi * freq * t) * np.exp(-decay * t)

    def boop(seconds, decay):                       # a pitch that drops: the "boop"
        n = int(seconds * rate)
        t = np.arange(n) / rate
        freq = 260.0 - 110.0 * np.minimum(t / seconds, 1.0)
        return np.sin(2 * np.pi * np.cumsum(freq) / rate) * np.exp(-decay * t)

    def place(parts):
        length = max(int(offset * rate) + len(part) for offset, part in parts)
        out = np.zeros(length)
        for offset, part in parts:
            start = int(offset * rate)
            out[start:start + len(part)] += part
        return out

    if name in ("hover", "select"):                 # a soft, quiet click (the same click for both)
        return place([(0.0, tone(1200, 0.02, 220)), (0.0, 0.3 * tone(2400, 0.01, 400))])
    if name in ("record_start", "record_stop"):     # marimba pair: start goes low then high, stop goes high then low
        high = place([(0.0, tone(587, 0.5, 6)), (0.0, 0.2 * tone(1761, 0.1, 40))])
        low = place([(0.0, tone(392, 0.3, 8)), (0.0, 0.2 * tone(1176, 0.1, 40))])
        if name == "record_start":
            return place([(0.0, low), (0.14, high)])
        return place([(0.0, high), (0.14, low)])
    raise ValueError(name)


def _wav_bytes(name, volume):
    raw = _sound_wave(name)
    peak = float(np.max(np.abs(raw))) or 1.0
    samples = raw / peak * _SOUND_LEVEL[name] * volume
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SOUND_RATE)
        out.writeframes(pcm)
    return buffer.getvalue()


def play_sound(name, volume_percent=None, force=False):
    """Play one sound without waiting for it. Silent when sounds are off, the volume is zero,
    or this isn't Windows. force=True plays it even when sounds are switched off (Test sound)."""
    if winsound is None:
        return
    if not force and not SETTINGS.get("sounds_on", True):
        return
    percent = SETTINGS.get("sound_volume", 10) if volume_percent is None else volume_percent
    if percent <= 0:
        return
    key = (name, int(percent))
    data = _SOUND_CACHE.get(key)
    if data is None:
        data = _SOUND_CACHE[key] = _wav_bytes(name, percent / 100.0)
    # SND_MEMORY can't be combined with SND_ASYNC (Windows then plays nothing), so the
    # blocking call runs on a short-lived helper thread instead. The UI never waits.
    threading.Thread(target=_play_wav_memory, args=(data,), daemon=True).start()


def _play_wav_memory(data):
    try:
        winsound.PlaySound(data, winsound.SND_MEMORY)
    except RuntimeError:
        pass


def play_hover():
    """Hover clicks are quiet and never closer together than 70 ms, so sweeping the mouse
    across the buttons doesn't turn into a rattle."""
    now = time.monotonic()
    if now - _hover_clock[0] < 0.07:
        return
    _hover_clock[0] = now
    play_sound("hover")


def as_source(value):
    """A source is ("desktop", output_index_or_None) or ("input", device_index).
    A bare number means an input device."""
    if isinstance(value, (tuple, list)) and len(value) == 2 and value[0] in ("desktop", "input"):
        return value[0], value[1]
    return "input", int(value)


class LiveTranscriber:
    """Captures audio, finds phrases, runs Whisper. Talks to the UI only by
    putting tuples on `events`:

        ("status", text)  ("listening", model_name)  ("level", rms)
        ("partial", utt_id, text)  ("final", utt_id, text, audio_seconds, took_seconds)
        ("error", message, details)  ("finished",)
    """

    def __init__(self, events):
        self.events = events
        self.model = None
        self.model_name = ""
        self.model_label = ""
        self.model_lock = threading.Lock()
        self.preloading = False
        self.stream = None
        self.stream_rate = SAMPLE_RATE
        self.fill_silence = False           # desktop audio sends nothing while the PC is silent
        self.stop_event = threading.Event()
        self.worker = None
        self.collector = None
        self.audio_q = queue.Queue(maxsize=300)
        self.utter_q = queue.Queue(maxsize=8)
        self.partial_lock = threading.Lock()
        self.partial_slot = None
        self.sink = None                    # the CallRecorder this side's audio goes into, or None
        self.channel = 0                    # which channel of that file this side fills
        self.gain = 1.0                     # recording gain only (not what Whisper hears)
        self.capture_only = False           # the microphone: recorded and metered, never transcribed
        self.infer_lock = threading.Lock()  # Whisper runs one call at a time, even when two sides share it

    @property
    def busy(self):
        return self.worker is not None and self.worker.is_alive()

    def start(self, source):
        if self.busy:
            return False
        self.stop_event = threading.Event()
        self.audio_q = queue.Queue(maxsize=300)
        self.utter_q = queue.Queue(maxsize=8)
        self.partial_slot = None
        self.worker = threading.Thread(
            target=self._worker, args=(source, self.stop_event), daemon=True)
        self.worker.start()
        return True

    def stop(self):
        self.stop_event.set()
        stream, self.stream = self.stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    # -- setup ---------------------------------------------------------------
    def preload(self):
        """Load the model in the background when the app opens, so Start Listening
        is instant instead of making the first call wait several seconds."""
        if self.model is not None or self.preloading:
            return
        self.preloading = True
        threading.Thread(target=self._preload, daemon=True).start()

    def _preload(self):
        try:
            self._load_model()
            self.events.put(("ready", self.model_label))
        except Exception as exc:
            self.events.put(("status", f"Could not preload the speech model ({exc}). "
                                       "It will be tried again when you click Start Listening."))
        finally:
            self.preloading = False

    def _make_model(self, name, device, compute):
        kwargs = {"device": device, "compute_type": compute}
        if device == "cpu":
            kwargs["cpu_threads"] = CPU_THREADS
        model = WhisperModel(local_model_dir(name) or name, **kwargs)
        # The very first call does one-off memory set-up and is much slower than the
        # rest. Do it now so the first real phrase isn't. The voice filter is off here
        # because it would skip pure silence and the model would never actually run.
        segments, _info = model.transcribe(
            np.zeros(SAMPLE_RATE * 2, dtype=np.float32),
            language="en", beam_size=1, vad_filter=False, without_timestamps=True)
        list(segments)                              # transcribe() is lazy: this runs it
        return model

    def _load_model(self):
        """Load Whisper once. The UI preloads it, and the lock makes a click on
        Start wait for that instead of loading a second copy.

        With an NVIDIA GPU the GPU is tried first for the requested model. If the GPU can't
        be used we carry on with the CPU and say so in the status text."""
        with self.model_lock:
            if self.model is not None:
                return
            cpu_compute = COMPUTE_TYPE if DEVICE == "cpu" else "int8"
            attempts = [(MODEL_SIZE, "cuda", COMPUTE_TYPE)] if DEVICE == "cuda" else []
            for name in dict.fromkeys([MODEL_SIZE, "base.en", "tiny.en"]):
                attempts.append((name, "cpu", cpu_compute))
            note = ""
            last_error = None
            for name, device, compute in attempts:
                where = "GPU" if device == "cuda" else "CPU"
                origin = "from the app's models folder" if local_model_dir(name) else "the first run downloads it"
                self.events.put(("status", f"Loading Whisper '{name}' on {where} ({origin})…"))
                try:
                    self.model = self._make_model(name, device, compute)
                except Exception as exc:
                    last_error = exc
                    if device == "cuda":
                        note = f" (GPU unavailable: {str(exc)[:80]})"
                    continue
                self.model_name = name
                self.model_label = f"{name} on {where}{note}"
                return
            raise RuntimeError(f"Could not load a Whisper model: {last_error}") from last_error

    def _open_stream(self, source):
        stream, rate, fill_silence = self._open_capture(source, self._audio_callback)
        self.stream, self.stream_rate, self.fill_silence = stream, rate, fill_silence

    def _open_capture(self, source, callback):
        """Open desktop audio or an input device. -> (stream, sample_rate, fill_silence)."""
        kind, index = as_source(source)
        if kind == "desktop":
            stream = DesktopAudioStream(index, callback)
            try:
                stream.start()
            except Exception as exc:
                stream.close()
                raise RuntimeError(f"Could not start desktop audio: {exc}") from exc
            return stream, stream.rate, True
        stream, rate = self._open_input(index, callback)
        return stream, rate, False

    def probe(self, source, seconds=3.0):
        """Listen for a few seconds and report (blocks_received, loudest_rms). Powers the
        Test Audio button. Windows sends no blocks at all while the PC is silent."""
        levels = []

        def listen(indata, frames, time_info, status):
            try:
                mono = indata.mean(axis=1) if indata.shape[1] > 1 else indata[:, 0]
                levels.append(float(np.sqrt(np.mean(np.square(mono)))))
            except Exception:
                pass

        stream, _rate, _fill = self._open_capture(source, listen)
        try:
            time.sleep(seconds)
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        return len(levels), (max(levels) if levels else 0.0)

    def _open_input(self, device_index, callback):
        """Open a microphone, headset or cable. Tries 16 kHz first, then the device's own
        rate (some drivers refuse 16 kHz) and converts in software. -> (stream, rate)."""
        if sd is None:
            raise RuntimeError("Microphone support (the sounddevice library) isn't available here. "
                               "Choose Desktop audio instead.")
        info = sd.query_devices(device_index)
        native_rate = int(info.get("default_samplerate") or 48000)
        channels = 2 if int(info.get("max_input_channels", 1)) >= 2 else 1
        attempts = [(SAMPLE_RATE, channels), (native_rate, channels)]
        if channels == 2:
            attempts += [(SAMPLE_RATE, 1), (native_rate, 1)]
        last_error = None
        for rate, ch in attempts:
            stream = None
            try:
                stream = sd.InputStream(
                    device=device_index, channels=ch, samplerate=rate,
                    blocksize=int(rate * BLOCK_SECONDS), dtype="float32",
                    callback=callback)
                stream.start()
                return stream, rate
            except Exception as exc:
                last_error = exc
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass
        raise RuntimeError(f"Could not open the selected audio input: {last_error}")

    def _audio_callback(self, indata, frames, time_info, status):
        # Runs on the audio driver's thread: keep it tiny.
        try:
            mono = indata.mean(axis=1) if indata.shape[1] > 1 else indata[:, 0].copy()
            mono = np.asarray(mono, dtype=np.float32)
            try:
                self.audio_q.put_nowait(mono)
            except queue.Full:
                try:
                    self.audio_q.get_nowait()          # drop the oldest, keep the newest
                except queue.Empty:
                    pass
                try:
                    self.audio_q.put_nowait(mono)
                except queue.Full:
                    pass
        except Exception:
            pass

    # -- threads ---------------------------------------------------------------
    def _push_utterance(self, item):
        try:
            self.utter_q.put_nowait(item)
        except queue.Full:
            try:
                self.utter_q.get_nowait()              # drop the oldest phrase
            except queue.Empty:
                pass
            self.events.put(("status",
                             "Transcription is behind and is dropping old audio. Use a smaller model "
                             "(MRAI_MODEL_SIZE=base.en), a GPU (MRAI_DEVICE=cuda) or MRAI_PARTIALS=0."))
            try:
                self.utter_q.put_nowait(item)
            except queue.Full:
                pass

    def _collector(self, stop, rate):
        """Cheap work only: resample, detect phrases, hand them to the worker."""
        segmenter = UtteranceSegmenter()
        audio_clock = 0.0
        next_partial = PARTIAL_INTERVAL_SECONDS
        last_data = time.monotonic()
        while not stop.is_set():
            try:
                raw = self.audio_q.get(timeout=0.1)
                last_data = time.monotonic()
            except queue.Empty:
                if not self.fill_silence or time.monotonic() - last_data < 0.25:
                    continue
                # Windows' desktop-audio capture sends nothing at all while the PC is quiet.
                # Without these blocks a finished sentence would never see its closing
                # silence and would sit unprocessed until the customer spoke again.
                raw = np.zeros(max(1, int(rate * BLOCK_SECONDS)), dtype=np.float32)
            gained = raw * np.float32(self.gain)
            sink = self.sink
            if sink is not None:
                try:
                    sink.feed(self.channel, to_16k(gained, rate))
                except Exception:
                    pass
            self.events.put(("rec_level", float(np.sqrt(np.mean(np.square(gained)))) if len(gained) else 0.0))
            if self.capture_only:
                continue
            block = to_16k(raw, rate)
            rms, finished = segmenter.feed(block)
            self.events.put(("level", rms))
            audio_clock += len(block) / SAMPLE_RATE
            for item in finished:
                with self.partial_lock:
                    self.partial_slot = None           # the final text replaces any live text
                self._push_utterance(item)
            if PARTIALS_ENABLED and segmenter.in_speech and audio_clock >= next_partial:
                snap = segmenter.snapshot()
                if snap is not None:
                    next_partial = audio_clock + PARTIAL_INTERVAL_SECONDS
                    with self.partial_lock:
                        self.partial_slot = snap       # overwrite: only the newest matters

    def _worker(self, source, stop):
        try:
            if not self.capture_only:
                self._load_model()                      # instant if the preload already finished
            if stop.is_set():
                return
            self._open_stream(source)
            self.collector = threading.Thread(
                target=self._collector, args=(stop, self.stream_rate), daemon=True)
            self.collector.start()
            self.events.put(("listening", self.model_label))

            while not stop.is_set():
                # Finished phrases always come first.
                try:
                    utt_id, audio = self.utter_q.get(timeout=0.05)
                except queue.Empty:
                    pass
                else:
                    self._run_final(utt_id, audio)
                    continue
                with self.partial_lock:
                    snap, self.partial_slot = self.partial_slot, None
                if snap is None:
                    continue
                utt_id, audio = snap
                text = self._transcribe(audio, final=False)
                if text and not stop.is_set():
                    self.events.put(("partial", utt_id, text))
        except Exception as exc:
            self.events.put(("error", str(exc), traceback.format_exc()))
        finally:
            self.stop()
            if self.collector is not None:
                self.collector.join(timeout=1.0)
            self.events.put(("finished",))

    def _run_final(self, utt_id, audio):
        if SAVE_DEBUG_AUDIO:
            self._save_debug_audio(utt_id, audio)
        started = time.monotonic()
        try:
            text = self._transcribe(audio, final=True)
        except Exception as exc:
            self.events.put(("status", f"Transcription error: {exc}"))
            return
        took = time.monotonic() - started
        self.events.put(("final", utt_id, text, len(audio) / SAMPLE_RATE, took))

    def _transcribe(self, audio, final):
        audio = prepare_audio(audio)
        options = dict(
            language="en",
            beam_size=FINAL_BEAM if final else 1,
            best_of=1,
            # One temperature, not a list: a list makes Whisper decode the phrase again
            # whenever it looks doubtful, which can double the wait. Doubtful text is
            # dropped by the checks below instead.
            temperature=0.0,
            without_timestamps=True,               # fewer tokens to generate
            condition_on_previous_text=False,
            # Whisper's own voice filter as a second line of defence, plus its
            # standard "this isn't really speech" thresholds.
            vad_filter=True,
            vad_parameters=dict(threshold=0.4, min_silence_duration_ms=400, speech_pad_ms=300),
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
        if USE_DOMAIN_PROMPT:
            options["initial_prompt"] = DOMAIN_PROMPT
        with self.infer_lock:                   # the two sides may share one model: one call at a time
            segments, _info = self.model.transcribe(audio, **options)
            parts = []
            for seg in segments:
                if seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0:
                    continue
                if seg.compression_ratio > 2.4:
                    continue
                piece = seg.text.strip()
                if piece:
                    parts.append(piece)
        text = " ".join(parts).strip()
        return "" if looks_like_hallucination(text) else text

    @staticmethod
    def _save_debug_audio(utt_id, audio):
        try:
            folder = os.path.join(DATA_DIR, "debug_audio")
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f"{time.strftime('%H%M%S')}_{utt_id}.wav")
            pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            with wave.open(path, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(SAMPLE_RATE)
                wav.writeframes(pcm.tobytes())
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Look and feel
# ----------------------------------------------------------------------------
# Every colour in the window comes from one of the themes below. To change the look,
# set DEFAULT_THEME to "midnight", "violet", "daylight" or "stage" (or start the app
# with MRAI_THEME=violet), or edit any colour. "scale" makes all text bigger or smaller.
DEFAULT_THEME = "midnight"
THEMES = {
    # Dark, teal accent.
    "midnight": dict(
        bg="#0E1117", surface="#151A23", surface2="#1C2331", border="#283244",
        text="#E6EAF2", muted="#8A94A8", accent="#2DD4BF", accent_hover="#5EEAD4",
        on_accent="#04201B", ok="#34D399", warn="#FBBF24", bad="#F87171",
        select="#14504A", scale=1.0,
        chips=["#2DD4BF", "#60A5FA", "#F472B6", "#FBBF24", "#A78BFA", "#FB923C"]),
    # Dark, violet accent.
    "violet": dict(
        bg="#11111A", surface="#181824", surface2="#20202F", border="#30304A",
        text="#ECECF6", muted="#9494B0", accent="#8B5CF6", accent_hover="#A78BFA",
        on_accent="#FFFFFF", ok="#4ADE80", warn="#FBBF24", bad="#FB7185",
        select="#3B2A6B", scale=1.0,
        chips=["#8B5CF6", "#38BDF8", "#F472B6", "#FBBF24", "#34D399", "#FB923C"]),
    # Light and clean, blue accent.
    "daylight": dict(
        bg="#F3F5F9", surface="#FFFFFF", surface2="#EEF1F6", border="#D9DFEA",
        text="#1A2233", muted="#667085", accent="#2563EB", accent_hover="#1D4ED8",
        on_accent="#FFFFFF", ok="#16A34A", warn="#D97706", bad="#DC2626",
        select="#CFE0FF", scale=1.0,
        chips=["#2563EB", "#0D9488", "#DB2777", "#D97706", "#7C3AED", "#EA580C"]),
    # Black, amber and extra-large text: built to be read at a glance on a live call.
    "stage": dict(
        bg="#000000", surface="#0B0B0B", surface2="#151515", border="#333333",
        text="#FFFFFF", muted="#B0B0B0", accent="#FFB000", accent_hover="#FFC845",
        on_accent="#000000", ok="#3DDC84", warn="#FFB000", bad="#FF5C5C",
        select="#5A3E00", scale=1.15,
        chips=["#FFB000", "#4DD0E1", "#FF80AB", "#B2FF59", "#B388FF", "#FF9E40"]),
}
# User settings (chosen on the Settings tab) live in settings.json in the data folder.
# The MRAI_THEME and MRAI_FONT environment variables still override them, for testing.
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")
DEFAULT_SETTINGS = {
    "theme": DEFAULT_THEME,         # a preset name, or the name of a theme saved on the Settings tab
    "text_size": 1.0,               # multiplies the text size of the main window
    "font": "Segoe UI",
    "colors": {},                   # colour changes on top of the theme, e.g. {"accent": "#FF8800"}
    "saved_themes": {},             # name -> {colour role: colour}
    "popup_text_size": 1.0,         # text size of the Suggested Rebuttal window
    "popup_on_top": True,           # keep the pop-up above other windows
    "popup_show_customer": True,    # show the "Customer said" section
    "popup_show_ask": True,         # show the "Ask next" section
    "sounds_on": True,              # click and recording sounds
    "sound_volume": 10,             # 0 to 100
}
TEXT_SIZES = [("Small", 0.9), ("Normal", 1.0), ("Large", 1.15), ("Extra large", 1.3)]
FONT_CHOICES = ["Segoe UI", "Arial", "Calibri", "Verdana", "Tahoma"]
COLOR_ROLES = [
    ("bg", "Window background"), ("surface", "Panels"), ("surface2", "Input boxes"),
    ("border", "Borders"), ("text", "Text"), ("muted", "Secondary text"),
    ("accent", "Buttons and highlights"), ("on_accent", "Text on buttons"),
    ("ok", "Listening light"), ("warn", "Warnings"), ("bad", "Errors"),
    ("select", "Selected text background"),
]
COLOR_ROLE_KEYS = [key for key, _ in COLOR_ROLES]
_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
_ENV_THEME = os.environ.get("MRAI_THEME", "").strip()


def _clean_settings(raw):
    """Keep only valid values. Anything wrong, missing or unknown falls back to the default."""
    clean = dict(DEFAULT_SETTINGS)
    clean["colors"] = {}
    clean["saved_themes"] = {}
    if not isinstance(raw, dict):
        return clean
    theme = raw.get("theme")
    if isinstance(theme, str) and theme.strip():
        clean["theme"] = theme.strip()
    font_name = raw.get("font")
    if isinstance(font_name, str) and font_name.strip():
        clean["font"] = font_name.strip()
    for key in ("text_size", "popup_text_size"):
        try:
            clean[key] = min(1.6, max(0.8, float(raw.get(key, DEFAULT_SETTINGS[key]))))
        except (TypeError, ValueError):
            pass
    for key in ("popup_on_top", "popup_show_customer", "popup_show_ask", "sounds_on"):
        if isinstance(raw.get(key), bool):
            clean[key] = raw[key]
    try:
        clean["sound_volume"] = min(100, max(0, int(raw.get("sound_volume", DEFAULT_SETTINGS["sound_volume"]))))
    except (TypeError, ValueError):
        pass
    colors = raw.get("colors")
    if isinstance(colors, dict):
        clean["colors"] = {key: value for key, value in colors.items()
                           if key in COLOR_ROLE_KEYS and isinstance(value, str) and _HEX_COLOR.match(value)}
    saved = raw.get("saved_themes")
    if isinstance(saved, dict):
        for name, colors in saved.items():
            if isinstance(name, str) and name.strip() and isinstance(colors, dict):
                good = {key: value for key, value in colors.items()
                        if key in COLOR_ROLE_KEYS and isinstance(value, str) and _HEX_COLOR.match(value)}
                if good:
                    clean["saved_themes"][name.strip()] = good
    return clean


def load_settings():
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as file:
            raw = json.load(file)
    except (OSError, ValueError):
        raw = None                                  # missing or unreadable: use the defaults
    return _clean_settings(raw)


def save_settings(settings):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as file:
            json.dump(_clean_settings(settings), file, indent=2)
    except OSError as exc:
        raise OSError(f"Could not save settings: {exc}") from exc


def build_theme(settings, theme_name=None):
    """The colours in use: the chosen theme, plus any colour changes, at the chosen text size.
    Always returns a fresh dict, so the presets are never changed."""
    name = theme_name or settings["theme"]
    if name in THEMES or name.lower() in THEMES:
        theme = dict(THEMES.get(name) or THEMES[name.lower()])
    elif name in settings["saved_themes"]:
        theme = dict(THEMES[DEFAULT_THEME])
        theme.update(settings["saved_themes"][name])
    else:
        theme = dict(THEMES[DEFAULT_THEME])
    base_scale = THEMES.get(name, THEMES.get(name.lower(), THEMES[DEFAULT_THEME])).get("scale", 1.0)
    theme.update(settings["colors"])
    theme["scale"] = base_scale * settings["text_size"]
    return theme


SETTINGS = load_settings()
THEME = build_theme(SETTINGS, _ENV_THEME or None)
THEME_NAME = _ENV_THEME or SETTINGS["theme"]
UI_FONT = os.environ.get("MRAI_FONT", "").strip() or SETTINGS["font"]
MONO_FONT = "Consolas" if sys.platform == "win32" else "DejaVu Sans Mono"
_UI_SCALE = 1.0                     # set from the screen's DPI when the window opens

# (colour name in the theme, label) for the status light in the header.
UI_STATES = {
    "idle": ("muted", "Idle"),
    "ready": ("accent", "Ready"),
    "loading": ("warn", "Loading"),
    "listening": ("ok", "Listening"),
    "error": ("bad", "Error"),
}


def _enable_dpi_awareness():
    """Windows: draw crisp text on high-DPI screens instead of letting Windows blow
    a blurry bitmap up. Set MRAI_DPI=0 to switch this off."""
    if sys.platform != "win32" or os.environ.get("MRAI_DPI") == "0":
        return
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def px(n):
    """Scale a pixel size for the screen's DPI."""
    return int(round(n * _UI_SCALE))


_FONT_SCALE_OVERRIDE = None         # set while the Suggested Rebuttal window is built


def font(size, style="normal"):
    scale = _FONT_SCALE_OVERRIDE if _FONT_SCALE_OVERRIDE is not None else THEME["scale"]
    return (UI_FONT, max(7, int(round(size * scale))), style)


def _rgb(color):
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def _luma(color):
    r, g, b = _rgb(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def blend(a, b, t):
    """Mix colour a towards colour b by the fraction t (0 to 1)."""
    ra, ga, ba = _rgb(a)
    rb, gb, bb = _rgb(b)
    return "#%02x%02x%02x" % (round(ra + (rb - ra) * t), round(ga + (gb - ga) * t), round(ba + (bb - ba) * t))


def text_on(color):
    """Black or white, whichever reads better on `color`."""
    return "#000000" if _luma(color) > 150 else "#FFFFFF"


def category_color(category):
    """A steady colour per objection category (the same category is always the same colour)."""
    chips = THEME["chips"]
    return chips[zlib.crc32(str(category).lower().encode("utf-8")) % len(chips)]


def rounded_rect(canvas, x1, y1, x2, y2, r, **kw):
    points = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2,
              x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
    return canvas.create_polygon(points, smooth=True, **kw)


def make_card(parent, title=None, pad=14):
    """A bordered panel with an optional small heading. Returns (outer, body):
    pack the outer frame, put the content in the body."""
    T = THEME
    outer = tk.Frame(parent, bg=T["surface"], highlightthickness=1,
                     highlightbackground=T["border"], highlightcolor=T["border"])
    if title:
        tk.Label(outer, text=title.upper(), bg=T["surface"], fg=T["muted"],
                 font=font(8, "bold")).pack(anchor="w", padx=px(pad), pady=(px(pad - 2), 0))
    body = tk.Frame(outer, bg=T["surface"])
    body.pack(fill="both", expand=True, padx=px(pad), pady=(px(6 if title else pad), px(pad)))
    return outer, body


def make_text(parent, height, size=10, bg_key="surface2", border=True, style="normal"):
    T = THEME
    return tk.Text(
        parent, height=height, wrap="word", bg=T[bg_key], fg=T["text"], insertbackground=T["text"],
        selectbackground=T["select"], selectforeground=T["text"],
        inactiveselectbackground=T["select"], relief="flat", bd=0,
        highlightthickness=1 if border else 0, highlightbackground=T["border"],
        highlightcolor=T["accent"], padx=px(10), pady=px(8), font=font(size, style),
        spacing1=2, spacing3=2, undo=True)


def make_entry(parent, variable=None):
    T = THEME
    return tk.Entry(
        parent, textvariable=variable, bg=T["surface2"], fg=T["text"], insertbackground=T["text"],
        selectbackground=T["select"], selectforeground=T["text"], relief="flat", bd=0,
        highlightthickness=1, highlightbackground=T["border"], highlightcolor=T["accent"],
        font=font(10))


def field_label(parent, text):
    return tk.Label(parent, text=text, bg=THEME["surface"], fg=THEME["muted"], font=font(9),
                    anchor="w", justify="left")


def wrap_to_width(label, margin=4):
    """Keep a label's text wrapping to whatever width the window gives it."""
    label.bind("<Configure>", lambda e: label.configure(wraplength=max(px(120), e.width - px(margin))))


class FlatButton(tk.Label):
    """A flat, coloured button that looks identical on every system (the native
    Windows button can't be recoloured). kind: "primary", "ghost" or "danger"."""

    def __init__(self, parent, text, command=None, kind="ghost", padx=16, **kw):
        self.kind = kind
        self.command = command
        self._enabled = True
        self._over = False
        super().__init__(parent, text=text, font=font(10, "bold"), padx=px(padx), pady=px(7),
                         bd=0, **kw)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonRelease-1>", self._on_release)
        self._paint()

    def _colors(self):
        T = THEME
        if not self._enabled:
            return T["surface2"], T["muted"], T["border"]
        if self.kind == "primary":
            fill = T["accent_hover"] if self._over else T["accent"]
            return fill, T["on_accent"], fill
        if self.kind == "danger":
            if self._over:
                return T["bad"], text_on(T["bad"]), T["bad"]
            return T["surface2"], T["bad"], T["border"]
        fill = blend(T["surface2"], T["text"], 0.10) if self._over else T["surface2"]
        return fill, T["text"], T["border"]

    def _paint(self):
        fill, ink, edge = self._colors()
        tk.Label.configure(self, bg=fill, fg=ink, highlightthickness=1, highlightbackground=edge,
                           highlightcolor=edge, cursor="hand2" if self._enabled else "arrow")

    def _on_enter(self, _event):
        self._over = True
        self._paint()
        if self._enabled:
            play_hover()

    def _on_leave(self, _event):
        self._over = False
        self._paint()

    def _on_release(self, _event):
        if self._enabled and self._over and self.command:
            play_sound("select")
            self.command()

    def configure(self, cnf=None, **kw):
        if "state" in kw:
            self._enabled = kw.pop("state") != "disabled"
        if "command" in kw:
            self.command = kw.pop("command")
        result = tk.Label.configure(self, cnf, **kw) if (cnf or kw) else None
        self._paint()
        return result

    config = configure


class LevelMeter(tk.Canvas):
    """A segmented input-level meter. Use .configure(value=0..100) like a progress bar."""

    SEGMENTS = 36

    def __init__(self, parent, **kw):
        super().__init__(parent, height=px(10), bg=THEME["surface"], highlightthickness=0, bd=0, **kw)
        self._value = 0.0
        self.bind("<Configure>", lambda _e: self._draw())

    def configure(self, cnf=None, **kw):
        redraw = False
        if "value" in kw:
            self._value = float(kw.pop("value"))
            redraw = True
        result = tk.Canvas.configure(self, cnf, **kw) if (cnf or kw) else None
        if redraw:
            self._draw()
        return result

    config = configure

    def _draw(self):
        self.delete("all")
        width, height = self.winfo_width(), self.winfo_height()
        if width <= 4:
            return
        T = THEME
        n, gap = self.SEGMENTS, 2
        seg = (width - gap * (n - 1)) / n
        lit = int(round(max(0.0, min(1.0, self._value / 100.0)) * n))
        for i in range(n):
            if i >= lit:
                fill = T["border"]
            elif i < n * 0.62:
                fill = T["ok"]
            elif i < n * 0.86:
                fill = T["warn"]
            else:
                fill = T["bad"]
            x = i * (seg + gap)
            self.create_rectangle(x, 0, x + seg, height, fill=fill, outline="")


class RecordDot(tk.Canvas):
    """A round white button with a bevel and a red dot in the middle. The dot blinks while
    recording. Clicking it calls command, which turns recording on or off."""

    RED, RED_OFF, GREY = "#E5322D", "#F3CFCD", "#9AA1A9"

    def __init__(self, parent, command=None, size=22):
        self.size = px(size)
        super().__init__(parent, width=self.size, height=self.size, bg=THEME["surface"], highlightthickness=0,
                         bd=0, cursor="hand2")
        self.command = command
        self.armed = True
        self.blinking = False
        self._dot_on = True
        self._job = None
        self.bind("<Button-1>", lambda _e: self.command() if self.command else None)
        self.bind("<Enter>", lambda _e: play_hover())
        self._draw()

    def set_state(self, armed, blinking):
        self.armed, self.blinking = armed, blinking
        if blinking and self._job is None:
            self._dot_on = True
            self._tick()
        elif not blinking:
            self._cancel()
            self._dot_on = True
        self._draw()

    def stop(self):
        """Call before the widget is destroyed, so the blink stops."""
        self._cancel()

    def _cancel(self):
        if self._job is not None:
            try:
                self.after_cancel(self._job)
            except tk.TclError:
                pass
            self._job = None

    def _tick(self):
        self._job = None
        self._dot_on = not self._dot_on
        self._draw()
        self._job = self.after(500, self._tick)

    def _draw(self):
        self.delete("all")
        size = self.size
        dark, light, face = "#8C949D", "#FFFFFF", "#F7F8FA"
        self.create_oval(1, 1, size - 1, size - 1, fill=dark, outline="")      # shadow, bottom and right
        self.create_oval(0, 0, size - 3, size - 3, fill=light, outline="")     # highlight, top and left
        self.create_oval(3, 3, size - 3, size - 3, fill=face, outline="")      # the white face
        radius = size * 0.2
        centre = size / 2
        if not self.armed:
            colour = self.GREY
        elif self.blinking and not self._dot_on:
            colour = self.RED_OFF
        else:
            colour = self.RED
        self.create_oval(centre - radius, centre - radius, centre + radius, centre + radius,
                         fill=colour, outline="")


class TabBar(tk.Frame):
    """Text tabs with an accent underline on the active one."""

    def __init__(self, parent, names, on_select):
        super().__init__(parent, bg=THEME["bg"])
        self.on_select = on_select
        self.current = None
        self.items = {}
        for name in names:
            cell = tk.Frame(self, bg=THEME["bg"])
            cell.pack(side="left", padx=(0, px(6)))
            label = tk.Label(cell, text=name, font=font(10, "bold"), bg=THEME["bg"], fg=THEME["muted"],
                             padx=px(12), pady=px(7), cursor="hand2")
            label.pack()
            underline = tk.Frame(cell, bg=THEME["bg"], height=px(2))
            underline.pack(fill="x")
            label.bind("<Button-1>", lambda _e, n=name: self._clicked(n))
            label.bind("<Enter>", lambda _e, n=name: self._hover(n, True))
            label.bind("<Leave>", lambda _e, n=name: self._hover(n, False))
            self.items[name] = (label, underline)

    def _clicked(self, name):
        play_sound("select")
        self.select(name)

    def _hover(self, name, over):
        if name != self.current:
            self.items[name][0].configure(fg=THEME["text"] if over else THEME["muted"])
            if over:
                play_hover()

    def select(self, name):
        for tab, (label, underline) in self.items.items():
            active = tab == name
            label.configure(fg=THEME["text"] if active else THEME["muted"])
            underline.configure(bg=THEME["accent"] if active else THEME["bg"])
        self.current = name
        self.on_select(name)


# ----------------------------------------------------------------------------
# The app
# ----------------------------------------------------------------------------
class MortgageRebuttalApp:
    def __init__(self, root):
        global _UI_SCALE
        self.root = root
        try:
            _UI_SCALE = max(1.0, root.winfo_fpixels("1i") / 96.0)
        except Exception:
            _UI_SCALE = 1.0
        self.root.title("Rebuttal Butler")
        set_window_icon(self.root)
        self._apply_root_geometry()

        self.events = queue.Queue()
        self.engine = LiveTranscriber(self.events)
        # "My side": a second transcriber for the microphone. It shares the caller's Whisper model
        # and lock, and starts once the model is ready (see _start_my_side).
        self.mine_events = queue.Queue()
        self.mine_engine = LiveTranscriber(self.mine_events)
        self.mine_engine.infer_lock = self.engine.infer_lock
        self.mine_engine.capture_only = True        # "My side" is recorded, not transcribed
        self.mine_sources = {}
        self.pending_mine_start = None
        self.recording_now = False
        self.recording_armed = True         # the round button: recording is on by default
        self.call_file = None               # the CallRecorder for the current recording
        self.gain_db = {"caller": 0.0, "mine": 0.0}
        self.gain_scales = {}
        self.rec_meters = {}
        self.recording_paths = []
        self.playback = None
        self.playback_volume = 100.0
        self._play_data = np.zeros((0, 1), dtype=np.float32)
        self._play_pos = 0
        self._watching = False
        self.recording_notice = ""
        self.saved_names = []
        self.is_listening = False
        self.listening_live = False
        self.last_sound_time = 0.0
        self.no_audio_warned = False
        self.current_matches = []
        self.current_match_index = 0
        self.current_customer_text = ""
        self.last_final_text = ""
        self.last_final_time = 0.0
        self.last_final_utt_id = 0
        self.last_partial_text = ""
        self.current_live_match_title = ""
        self.ui_state = "loading"
        self._pulse_job = None
        self._pulse_on = False
        self.notations = []                         # one entry per highlight: tag, note, color
        self.highlight_count = 0
        self.tutorial_win = None
        self.tour_targets = {}                      # widgets the tour points at, filled in while building
        self._tour_glow = []                        # frames that make up the glow
        self._tour_corner = None                    # which glow corner the card sits on
        self._tour_target_widget = None

        self.device_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Choose what to listen to, then click Start Listening.")
        self.sources = {}
        self.listening_source_kind = "input"
        self.testing_audio = False
        self.search_var = tk.StringVar()

        self._apply_ttk_theme()
        self._build_main_window()
        self._build_response_window()
        self.refresh_devices()
        self._style_titlebar(self.root)
        self._style_titlebar(self.response_window)
        self.root.protocol("WM_DELETE_WINDOW", self.close_app)
        self.root.bind("<Configure>", self._on_tour_configure, add="+")
        self.root.after(50, self._poll_events)
        self._set_state("loading")
        self.engine.preload()                       # load Whisper now, not on the first click

    def _apply_ttk_theme(self):
        """Colour the few built-in (ttk) widgets we still use: the device drop-down and scrollbars."""
        T = THEME
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Mod.TCombobox", fieldbackground=T["surface2"], background=T["surface2"],
                        foreground=T["text"], arrowcolor=T["muted"], bordercolor=T["border"],
                        lightcolor=T["border"], darkcolor=T["border"], selectbackground=T["surface2"],
                        selectforeground=T["text"], padding=px(6))
        style.map("Mod.TCombobox",
                  fieldbackground=[("readonly", T["surface2"])],
                  foreground=[("readonly", T["text"])],
                  selectbackground=[("readonly", T["surface2"])],
                  selectforeground=[("readonly", T["text"])],
                  bordercolor=[("focus", T["accent"])],
                  lightcolor=[("focus", T["accent"])],
                  darkcolor=[("focus", T["accent"])],
                  arrowcolor=[("active", T["accent"])],
                  background=[("active", T["surface2"])])
        thumb = blend(T["border"], T["muted"], 0.30)
        style.layout("Mod.Vertical.TScrollbar", [("Vertical.Scrollbar.trough", {
            "sticky": "ns", "children": [("Vertical.Scrollbar.thumb", {"expand": "1", "sticky": "nswe"})]})])
        style.configure("Mod.Vertical.TScrollbar", troughcolor=T["surface"], background=thumb,
                        bordercolor=T["surface"], lightcolor=thumb, darkcolor=thumb,
                        arrowsize=px(11), gripcount=0, relief="flat")
        style.map("Mod.Vertical.TScrollbar", background=[("active", T["muted"]), ("pressed", T["accent"])],
                  lightcolor=[("active", T["muted"]), ("pressed", T["accent"])],
                  darkcolor=[("active", T["muted"]), ("pressed", T["accent"])])
        self.root.option_add("*TCombobox*Listbox.background", T["surface2"])
        self.root.option_add("*TCombobox*Listbox.foreground", T["text"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", T["accent"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", T["on_accent"])
        self.root.option_add("*TCombobox*Listbox.borderWidth", 0)

    def _style_titlebar(self, window):
        """Windows 10/11 only: a dark title bar for dark themes, and on Windows 11 a title bar
        in the theme's own colour. Purely cosmetic, so any problem is ignored."""
        if sys.platform != "win32":
            return
        try:
            import ctypes
            window.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
            dwm = ctypes.windll.dwmapi
            dark = ctypes.c_int(1 if _luma(THEME["bg"]) < 128 else 0)
            for attribute in (20, 19):              # 20: Windows 10 (2004+) and 11, 19: older Windows 10
                if dwm.DwmSetWindowAttribute(hwnd, attribute, ctypes.byref(dark), ctypes.sizeof(dark)) == 0:
                    break
            r, g, b = _rgb(THEME["bg"])
            caption = ctypes.c_int(r | (g << 8) | (b << 16))      # COLORREF is 0x00BBGGRR
            dwm.DwmSetWindowAttribute(hwnd, 35, ctypes.byref(caption), ctypes.sizeof(caption))   # Windows 11
        except Exception:
            pass

    def _set_state(self, state):
        """Update the status light in the header: idle, ready, loading, listening or error."""
        key, label = UI_STATES[state]
        self.ui_state = state
        color = THEME[key]
        self.state_dot.itemconfigure(self.state_dot_item, fill=color)
        self.state_label.configure(text=label.upper(), fg=color)
        if state == "listening" and self._pulse_job is None:
            self._pulse_job = self.root.after(650, self._pulse)

    def _pulse(self):
        if self.ui_state != "listening":
            self._pulse_job = None
            return
        self._pulse_on = not self._pulse_on
        color = THEME["ok"] if self._pulse_on else blend(THEME["ok"], THEME["bg"], 0.55)
        self.state_dot.itemconfigure(self.state_dot_item, fill=color)
        self._pulse_job = self.root.after(650, self._pulse)

    def _show_page(self, name):
        for page in self.pages.values():
            page.pack_forget()
        self.pages[name].pack(fill="both", expand=True)

    def _show_waiting(self):
        T = THEME
        self.match_chip.configure(text="READY", bg=T["border"], fg=T["muted"])
        self.match_counter.configure(text="")
        self.match_title.configure(text="Waiting for an objection…")
        self._set_text_widget(self.customer_quote, "")
        self._set_text_widget(self.response_text,
                              "The suggested rebuttal will appear here as soon as the customer raises an objection.")
        self.followup_label.configure(text="")

    # -- UI construction -------------------------------------------------------
    def _build_main_window(self):
        T = THEME
        header = tk.Frame(self.root, bg=T["bg"])
        header.pack(fill="x", padx=px(22), pady=(px(18), 0))

        logo = tk.Canvas(header, width=px(38), height=px(38), bg=T["bg"], highlightthickness=0)
        logo.pack(side="left")
        rounded_rect(logo, 1, 1, px(37), px(37), px(10), fill=T["accent"], outline="")
        for i, half in enumerate((5, 9, 14, 9, 5)):           # a little sound wave
            x = px(10) + i * px(4.5)
            logo.create_line(x, px(19) - px(half), x, px(19) + px(half), width=px(2.6),
                             capstyle="round", fill=T["on_accent"])

        titles = tk.Frame(header, bg=T["bg"])
        titles.pack(side="left", padx=(px(12), 0))
        tk.Label(titles, text="Rebuttal Butler", font=font(17, "bold"), bg=T["bg"],
                 fg=T["text"]).pack(anchor="w")
        tk.Label(titles, text="Live transcription  ·  objection matching  ·  suggested responses",
                 font=font(9), bg=T["bg"], fg=T["muted"]).pack(anchor="w")

        # Packed right-to-left: the status light goes in first so it sits at the far right,
        # and the logo is packed after it so it sits just to the left of the light.
        light = tk.Frame(header, bg=T["bg"])
        light.pack(side="right")
        self.state_dot = tk.Canvas(light, width=px(12), height=px(12), bg=T["bg"], highlightthickness=0)
        self.state_dot.pack(side="left", padx=(0, px(7)))
        self.state_dot_item = self.state_dot.create_oval(1, 1, px(11), px(11), fill=T["muted"], outline="")
        self.state_label = tk.Label(light, text="LOADING", font=font(9, "bold"), bg=T["bg"], fg=T["muted"])
        self.state_label.pack(side="left")

        self._main_logo = load_logo(px(100))
        if self._main_logo is not None:
            tk.Label(header, image=self._main_logo, bg=T["bg"], bd=0).pack(side="right", padx=(0, px(16)))

        # The round record button and the side dropdown, left of the logo (no extra height).
        recording = tk.Frame(header, bg=T["bg"])
        recording.pack(side="right", padx=(0, px(18)))
        self.record_dot = RecordDot(recording, command=self.toggle_recording, size=26)
        self.record_dot.pack(side="left", padx=(0, px(8)))
        tk.Label(recording, text="Record:", font=font(9), bg=T["bg"], fg=T["muted"]).pack(side="left", padx=(0, px(6)))
        self.record_var = tk.StringVar(value="Both")
        self.record_combo = ttk.Combobox(recording, textvariable=self.record_var,
                                         values=[label for label, _ in RECORD_CHOICES],
                                         state="readonly", style="Mod.TCombobox", font=font(10), width=10)
        self.record_combo.pack(side="left")
        self._refresh_record_dot()

        names = ["Live Assistant", "Add Rebuttal", "Library", "Script", "Recordings", "Settings"]
        self.tabbar = TabBar(self.root, names, self._show_page)
        self.tabbar.pack(fill="x", padx=px(22), pady=(px(14), 0))
        tk.Frame(self.root, bg=T["border"], height=1).pack(fill="x", padx=px(22))
        body = tk.Frame(self.root, bg=T["bg"])
        body.pack(fill="both", expand=True, padx=px(22), pady=px(14))
        self.pages = {name: tk.Frame(body, bg=T["bg"]) for name in names}

        # ---- Live Assistant ----
        live = self.pages["Live Assistant"]
        card, inner = make_card(live, "Listening")
        card.pack(fill="x", pady=(0, px(10)))
        controls = tk.Frame(inner, bg=T["surface"])
        controls.pack(fill="x", pady=(px(12), 0))
        self.tour_targets["start"] = controls
        self.start_button = FlatButton(controls, "Start Listening", self.start_listening, kind="primary")
        self.start_button.pack(side="left")
        self.stop_button = FlatButton(controls, "Stop Listening", self.stop_listening)
        self.stop_button.configure(state="disabled")
        self.stop_button.pack(side="left", padx=(px(8), 0))
        self.status_label = tk.Label(controls, textvariable=self.status_var, bg=T["surface"], fg=T["muted"],
                                     font=font(9), anchor="w", justify="left")
        self.status_label.pack(side="left", fill="x", expand=True, padx=px(14))
        wrap_to_width(self.status_label)

        meter_row = tk.Frame(inner, bg=T["surface"])
        meter_row.pack(fill="x", pady=(px(12), 0))
        tk.Label(meter_row, text="INPUT LEVEL", font=font(8, "bold"), bg=T["surface"],
                 fg=T["muted"]).pack(side="left", padx=(0, px(10)))
        self.level_bar = LevelMeter(meter_row)
        self.level_bar.pack(side="left", fill="x", expand=True)

        card, inner = make_card(live, "Latest recognized speech")
        card.pack(fill="x", pady=(0, px(10)))
        self.tour_targets["preview"] = card
        self.preview_text = make_text(inner, 3, size=12, bg_key="surface", border=False)
        self.preview_text.pack(fill="x")
        self.preview_text.insert("1.0", "Recognized speech will appear here after you start listening.")
        self.preview_text.configure(state="disabled")

        # The search bar is packed first (at the bottom) so a short window squeezes the
        # transcript, never the controls.
        search_card, inner = make_card(live, "Find a rebuttal manually")
        search_card.pack(side="bottom", fill="x")
        self.tour_targets["manual"] = search_card
        row = tk.Frame(inner, bg=T["surface"])
        row.pack(fill="x")
        search_entry = make_entry(row, self.search_var)
        search_entry.pack(side="left", fill="x", expand=True, padx=(0, px(10)), ipady=px(6))
        search_entry.bind("<Return>", lambda _event: self.manual_search())
        FlatButton(row, "Search", self.manual_search, kind="primary").pack(side="left")

        card, inner = make_card(live, "Transcript")
        card.pack(fill="both", expand=True, pady=(0, px(10)))
        # One toolbar row, packed bottom-first so it keeps its space in a short window.
        tools = tk.Frame(inner, bg=T["surface"])
        tools.pack(side="bottom", fill="x", pady=(px(8), 0))
        self.tour_targets["highlight"] = tools
        save_button = FlatButton(tools, "Save Transcript", self.save_transcript, kind="primary")
        save_button.pack(side="right")
        self.tour_targets["save"] = save_button
        FlatButton(tools, "Remove Highlight", self.remove_highlight).pack(side="right", padx=(0, px(8)))
        FlatButton(tools, "Highlight Selection", self.highlight_selection, kind="primary").pack(side="right", padx=(0, px(8)))
        self.color_var = tk.StringVar(value=HIGHLIGHT_COLORS[0][0])
        ttk.Combobox(tools, textvariable=self.color_var, values=[name for name, _ in HIGHLIGHT_COLORS],
                     state="readonly", style="Mod.TCombobox", font=font(10), width=8).pack(side="left", padx=(0, px(8)))
        self.note_var = tk.StringVar(value=NOTE_CHOICES[0])
        ttk.Combobox(tools, textvariable=self.note_var, values=NOTE_CHOICES, style="Mod.TCombobox",
                     font=font(10), width=18).pack(side="left", padx=(0, px(8)))
        tk.Label(tools, text="Highlight:", font=font(9), bg=T["surface"], fg=T["muted"]).pack(side="left", padx=(0, px(8)))
        wrap = tk.Frame(inner, bg=T["surface"])
        wrap.pack(fill="both", expand=True)
        self.history_text = make_text(wrap, 6, size=10, bg_key="surface", border=False)
        self.history_text.tag_configure("ts", foreground=T["muted"], font=(MONO_FONT, max(8, round(9 * T["scale"]))))
        self.history_text.tag_configure("fb", foreground=T["accent"], font=font(10, "bold"))
        history_scroll = ttk.Scrollbar(wrap, orient="vertical", command=self.history_text.yview,
                                       style="Mod.Vertical.TScrollbar")
        self.history_text.configure(yscrollcommand=history_scroll.set)
        self.history_text.pack(side="left", fill="both", expand=True)
        history_scroll.pack(side="right", fill="y")

        # ---- Add Rebuttal (the saved response is displayed verbatim) ----
        card, inner = make_card(self.pages["Add Rebuttal"], None, pad=18)
        card.pack(fill="both", expand=True)
        tk.Label(inner, text="Add a Rebuttal", font=font(15, "bold"), bg=T["surface"], fg=T["text"]).pack(anchor="w")
        intro = field_label(inner, "Enter the objection, alternate ways the customer may say it, and your approved "
                                   "script response. The response is stored exactly as typed.")
        intro.pack(fill="x", pady=(px(2), px(12)))
        wrap_to_width(intro)

        field_label(inner, "Objection / short title").pack(anchor="w")
        self.new_title_var = tk.StringVar()
        make_entry(inner, self.new_title_var).pack(fill="x", pady=(px(3), px(10)), ipady=px(6))

        field_label(inner, "Alternate ways the customer might say it (one per line; aim for 10)").pack(anchor="w")
        self.new_triggers_text = make_text(inner, 5)
        self.new_triggers_text.pack(fill="x", pady=(px(3), px(10)))
        self.new_triggers_text.insert("1.0", "I already have this\nI am already covered\nI do not need this")

        field_label(inner, "Approved rebuttal script (paste your wording here; it will not be rewritten)").pack(anchor="w")
        # Items at the bottom are packed first so they keep their space in a short window;
        # the script box (packed last) takes whatever is left.
        note = field_label(inner, "New entries are saved in rebuttal_library.json next to main.py, so they remain after closing the app.")
        note.pack(side="bottom", fill="x", pady=(px(10), 0))
        wrap_to_width(note)
        form_buttons = tk.Frame(inner, bg=T["surface"])
        form_buttons.pack(side="bottom", fill="x")
        FlatButton(form_buttons, "Accept & Add to Library", self.accept_new_rebuttal, kind="primary").pack(side="left")
        FlatButton(form_buttons, "Clear Form", self.clear_rebuttal_form).pack(side="left", padx=(px(8), 0))
        self.new_followup_var = tk.StringVar()
        make_entry(inner, self.new_followup_var).pack(side="bottom", fill="x", pady=(px(3), px(12)), ipady=px(6))
        field_label(inner, "Optional follow-up question").pack(side="bottom", anchor="w")
        self.new_response_text = make_text(inner, 4)
        self.new_response_text.pack(fill="both", expand=True, pady=(px(3), px(10)))

        self._build_library_tab(self.pages["Library"])
        self._build_script_tab(self.pages["Script"])
        self._build_recordings_tab(self.pages["Recordings"])
        self._build_settings_tab(self.pages["Settings"])
        self.tabbar.select("Live Assistant")

    # -- Settings and live apply ----------------------------------------------------
    def _apply_root_geometry(self):
        grow = THEME["scale"]                       # bigger-text settings get a proportionally bigger window
        screen_w, screen_h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        width = min(px(940 * grow), screen_w - px(40))      # never taller or wider than the screen
        height = min(px(840 * grow), screen_h - px(110))
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(min(px(780 * grow), width), min(px(640 * grow), height))
        self.root.configure(bg=THEME["bg"])

    def _sync_buttons(self):
        busy = self.is_listening or self.engine.busy or self.mine_engine.busy
        self.start_button.configure(state="disabled" if busy else "normal")
        self.stop_button.configure(state="normal" if self.is_listening else "disabled")

    def _restore_transcript(self, dump, widget=None):
        """Put the transcript back exactly as it was, including timestamps and highlights."""
        widget = widget or self.history_text
        for note in self.notations:
            color = dict(HIGHLIGHT_COLORS).get(note["color"], HIGHLIGHT_COLORS[0][1])
            widget.tag_configure(note["tag"], background=color, foreground="#101418")
        active = set()
        for key, value, _index in dump:
            if key == "tagon":
                active.add(value)
            elif key == "tagoff":
                active.discard(value)
            elif key == "text":
                widget.insert("end", value, tuple(sorted(tag for tag in active if tag != "sel")))
        widget.see("end")

    def _rebuild_ui(self):
        """Build every window again with the current settings. Transcript, highlights, the
        device list, listening state, the open pop-up and anything being typed are kept."""
        self._script_sync()
        keep = {
            "transcript": self.history_text.dump("1.0", "end-1c", text=True, tag=True),
            "record": self.record_var.get(),
            "armed": self.recording_armed,
            "mine": self.mine_var.get(),
            "mine_devices": list(self.mine_combo.cget("values")),
            "preview": self.preview_text.get("1.0", "end-1c"),
            "tab": self.tabbar.current,
            "devices": list(self.device_combo.cget("values")),
            "note": self.note_var.get(),
            "color": self.color_var.get(),
            "new_title": self.new_title_var.get(),
            "new_followup": self.new_followup_var.get(),
            "new_triggers": self.new_triggers_text.get("1.0", "end-1c"),
            "new_response": self.new_response_text.get("1.0", "end-1c"),
            "lib_title": self.library_title_var.get(),
            "lib_category": self.library_category_var.get(),
            "lib_triggers": self.library_triggers_text.get("1.0", "end-1c"),
            "lib_followup": self.library_followup_text.get("1.0", "end-1c"),
            "lib_response": self.library_response_text.get("1.0", "end-1c"),
            "lib_index": self.library_selected_index,
            "lib_is_new": self.library_is_new,
            "popup_state": self.response_window.state(),
            "popup_xy": (self.response_window.winfo_x(), self.response_window.winfo_y()),
        }
        if self.tutorial_win is not None:
            self.tutorial_win.destroy()
            self.tutorial_win = None
        self._clear_glow()
        self.record_dot.stop()
        for child in list(self.root.winfo_children()):
            child.destroy()

        self._apply_ttk_theme()
        self._apply_root_geometry()
        self.tour_targets = {}
        self._build_main_window()
        self._build_response_window()

        self.device_combo.configure(values=keep["devices"])
        self._restore_transcript(keep["transcript"])
        self.mine_combo.configure(values=keep["mine_devices"])
        self.mine_var.set(keep["mine"])
        self.record_var.set(keep["record"])
        self.recording_armed = keep["armed"]
        self._refresh_record_dot()
        self._set_preview(keep["preview"])
        self.note_var.set(keep["note"])
        self.color_var.set(keep["color"])
        self.new_title_var.set(keep["new_title"])
        self.new_followup_var.set(keep["new_followup"])
        self._replace_text(self.new_triggers_text, keep["new_triggers"].rstrip("\n"))
        self._replace_text(self.new_response_text, keep["new_response"].rstrip("\n"))
        if keep["lib_index"] is not None:
            self.refresh_library_list(keep["lib_index"])
        self.library_selected_index = keep["lib_index"]
        self.library_is_new = keep["lib_is_new"]
        self.library_title_var.set(keep["lib_title"])
        self.library_category_var.set(keep["lib_category"])
        self._replace_text(self.library_triggers_text, keep["lib_triggers"].rstrip("\n"))
        self._replace_text(self.library_followup_text, keep["lib_followup"].rstrip("\n"))
        self._replace_text(self.library_response_text, keep["lib_response"].rstrip("\n"))
        if self.current_matches:
            self.display_current_match()
        self._set_state(self.ui_state)
        self._sync_buttons()
        self._style_titlebar(self.root)
        self._style_titlebar(self.response_window)
        x, y = keep["popup_xy"]
        self.response_window.geometry(f"+{x}+{y}")
        if keep["popup_state"] == "withdrawn":
            self.response_window.withdraw()
        self.tabbar.select(keep["tab"] or "Live Assistant")

    def _build_settings_tab(self, parent):
        T = THEME
        self.draft = json.loads(json.dumps(SETTINGS))   # changes are staged here until Apply changes
        card, inner = make_card(parent, None, pad=18)
        card.pack(fill="both", expand=True)
        tk.Label(inner, text="Settings", font=font(15, "bold"), bg=T["surface"], fg=T["text"]).pack(anchor="w")
        intro = field_label(inner, "Make your changes, then click Apply changes. Your transcript, highlights "
                                   "and any open pop-up are kept.")
        intro.pack(fill="x", pady=(px(2), px(10)))
        wrap_to_width(intro)

        # Bottom rows first, so they keep their space in a short window.
        status_row = tk.Frame(inner, bg=T["surface"])
        status_row.pack(side="bottom", fill="x", pady=(px(8), 0))
        self.settings_status = tk.Label(status_row, text="", font=font(9), bg=T["surface"], fg=T["muted"],
                                        anchor="w", justify="left")
        self.settings_status.pack(fill="x")
        wrap_to_width(self.settings_status)
        save_row = tk.Frame(inner, bg=T["surface"])
        save_row.pack(side="bottom", fill="x", pady=(px(8), 0))
        field_label(save_row, "Save these colors as a theme named:").pack(side="left", padx=(0, px(8)))
        self.theme_name_var = tk.StringVar()
        make_entry(save_row, self.theme_name_var).pack(side="left", ipady=px(4), padx=(0, px(8)))
        FlatButton(save_row, "Save as theme", self.save_as_theme).pack(side="left")
        buttons = tk.Frame(inner, bg=T["surface"])
        buttons.pack(side="bottom", fill="x", pady=(px(8), 0))
        FlatButton(buttons, "Apply changes", self.apply_draft, kind="primary").pack(side="left")
        FlatButton(buttons, "Export settings", self.export_settings).pack(side="right")
        FlatButton(buttons, "Import settings", self.import_settings).pack(side="right", padx=(0, px(8)))

        settings_inner = inner
        card, audio_inner = make_card(inner, "Audio sources", pad=14)
        card.pack(fill="x", pady=(0, px(10)))
        self.tour_targets["audio"] = card
        row = tk.Frame(audio_inner, bg=T["surface"])
        row.pack(fill="x")
        self.device_combo = ttk.Combobox(row, textvariable=self.device_var, state="readonly",
                                         style="Mod.TCombobox", font=font(10))
        self.device_combo.pack(side="left", fill="x", expand=True, padx=(0, px(10)))
        FlatButton(row, "Refresh Devices", self.refresh_devices).pack(side="left")
        self.test_button = FlatButton(row, "Test Audio", self.test_audio)
        self.test_button.pack(side="left", padx=(px(8), 0))
        FlatButton(row, "Replay Tutorial", self.start_tutorial).pack(side="left", padx=(px(8), 0))
        mine_row = tk.Frame(audio_inner, bg=T["surface"])
        mine_row.pack(fill="x", pady=(px(8), 0))
        tk.Label(mine_row, text="My microphone:", font=font(10), bg=T["surface"], fg=T["text"]).pack(
            side="left", padx=(0, px(10)))
        self.mine_var = tk.StringVar()
        self.mine_combo = ttk.Combobox(mine_row, textvariable=self.mine_var, state="readonly",
                                       style="Mod.TCombobox", font=font(10))
        self.mine_combo.pack(side="left", fill="x", expand=True)
        hint = field_label(audio_inner, "Desktop audio hears whatever this PC plays, such as your browser dialer, "
                                  "through your speakers or headset. Nothing else to install, and your own "
                                  "microphone is picked above and recorded into the same file as the call.")
        hint.pack(fill="x", pady=(px(8), 0))
        wrap_to_width(hint)

        columns = tk.Frame(inner, bg=T["surface"])
        columns.pack(fill="both", expand=True)
        left = tk.Frame(columns, bg=T["surface"])
        left.pack(side="left", fill="both", expand=True, padx=(0, px(18)))
        right = tk.Frame(columns, bg=T["surface"])
        right.pack(side="left", fill="both", expand=True)

        # Left column: look and pop-up options
        tk.Label(left, text="LOOK", font=font(8, "bold"), bg=T["surface"], fg=T["muted"]).pack(anchor="w")
        field_label(left, "Theme").pack(anchor="w", pady=(px(6), 0))
        self.theme_choices = self._theme_choices()
        self.theme_combo = ttk.Combobox(left, values=list(self.theme_choices), state="readonly",
                                        style="Mod.TCombobox", font=font(10), width=22)
        self.theme_combo.set(self._theme_display(self.draft["theme"]))
        self.theme_combo.pack(anchor="w", pady=(px(3), px(8)))
        self.theme_combo.bind("<<ComboboxSelected>>", lambda _e: self._draft_theme_changed())
        field_label(left, "Text size").pack(anchor="w")
        self.size_combo = ttk.Combobox(left, values=[label for label, _ in TEXT_SIZES], state="readonly",
                                       style="Mod.TCombobox", font=font(10), width=22)
        self.size_combo.set(next((label for label, value in TEXT_SIZES
                                  if abs(value - self.draft["text_size"]) < 0.01), "Normal"))
        self.size_combo.pack(anchor="w", pady=(px(3), px(8)))
        field_label(left, "Font (or type the name of any font installed on this PC)").pack(anchor="w")
        self.font_combo = ttk.Combobox(left, values=FONT_CHOICES, style="Mod.TCombobox", font=font(10), width=22)
        self.font_combo.set(self.draft["font"])
        self.font_combo.pack(anchor="w", pady=(px(3), px(8)))

        tk.Label(left, text="SUGGESTED REBUTTAL WINDOW", font=font(8, "bold"), bg=T["surface"],
                 fg=T["muted"]).pack(anchor="w", pady=(px(14), 0))
        field_label(left, "Text size").pack(anchor="w", pady=(px(6), 0))
        self.popup_size_combo = ttk.Combobox(left, values=[label for label, _ in TEXT_SIZES], state="readonly",
                                             style="Mod.TCombobox", font=font(10), width=22)
        self.popup_size_combo.set(next((label for label, value in TEXT_SIZES
                                        if abs(value - self.draft["popup_text_size"]) < 0.01), "Normal"))
        self.popup_size_combo.pack(anchor="w", pady=(px(3), px(8)))
        self.on_top_var = tk.BooleanVar(value=bool(self.draft["popup_on_top"]))
        self.show_customer_var = tk.BooleanVar(value=bool(self.draft["popup_show_customer"]))
        self.show_ask_var = tk.BooleanVar(value=bool(self.draft["popup_show_ask"]))
        for text, variable in (("Keep on top of other windows", self.on_top_var),
                               ("Show the CUSTOMER SAID section", self.show_customer_var),
                               ("Show the ASK NEXT section", self.show_ask_var)):
            tk.Checkbutton(left, text=text, variable=variable, bg=T["surface"], fg=T["text"],
                           selectcolor=T["surface2"], activebackground=T["surface"],
                           activeforeground=T["text"], highlightthickness=0, font=font(10)).pack(anchor="w", pady=2)

        tk.Label(left, text="SOUNDS", font=font(8, "bold"), bg=T["surface"], fg=T["muted"]).pack(
            anchor="w", pady=(px(14), 0))
        self.sounds_var = tk.BooleanVar(value=bool(self.draft["sounds_on"]))
        tk.Checkbutton(left, text="Play click and recording sounds", variable=self.sounds_var, bg=T["surface"],
                       fg=T["text"], selectcolor=T["surface2"], activebackground=T["surface"],
                       activeforeground=T["text"], highlightthickness=0, font=font(10)).pack(anchor="w", pady=2)
        field_label(left, "Volume").pack(anchor="w", pady=(px(6), 0))
        self.volume_scale = tk.Scale(left, from_=0, to=100, orient="horizontal", length=px(220),
                                     bg=T["surface"], fg=T["text"], troughcolor=T["surface2"],
                                     highlightthickness=0, bd=0, activebackground=T["accent"])
        self.volume_scale.set(int(self.draft["sound_volume"]))
        self.volume_scale.pack(anchor="w")
        FlatButton(left, "Play a test sound", self._test_sound).pack(anchor="w", pady=(px(6), 0))

        # Right column: colors
        tk.Label(right, text="COLORS (click a swatch to change it)", font=font(8, "bold"), bg=T["surface"],
                 fg=T["muted"]).pack(anchor="w")
        grid = tk.Frame(right, bg=T["surface"])
        grid.pack(anchor="w", pady=(px(6), 0))
        self.swatches = {}
        for row, (role, label) in enumerate(COLOR_ROLES):
            tk.Label(grid, text=label, font=font(10), bg=T["surface"], fg=T["text"], anchor="w",
                     width=24).grid(row=row, column=0, sticky="w", pady=2)
            swatch = tk.Label(grid, width=8, bd=0, cursor="hand2", highlightthickness=1,
                              highlightbackground=T["border"])
            swatch.grid(row=row, column=1, sticky="w", padx=(px(8), 0), pady=2)
            swatch.bind("<Button-1>", lambda _e, key=role: self._pick_color(key))
            hex_label = tk.Label(grid, font=font(9), bg=T["surface"], fg=T["muted"], anchor="w", width=9)
            hex_label.grid(row=row, column=2, sticky="w", padx=(px(8), 0))
            self.swatches[role] = (swatch, hex_label)
        FlatButton(right, "Reset colors to the theme", self._reset_colors).pack(anchor="w", pady=(px(10), 0))
        self._refresh_settings_preview()

    def _theme_choices(self):
        choices = {key.title(): key for key in THEMES}
        for name in SETTINGS["saved_themes"]:
            choices[name] = name
        return choices

    def _theme_display(self, key):
        for display, value in self.theme_choices.items():
            if value == key:
                return display
        return key

    def _draft_theme_changed(self):
        self.draft["theme"] = self.theme_choices.get(self.theme_combo.get(), self.draft["theme"])
        self._refresh_settings_preview()

    def _refresh_settings_preview(self):
        preview = build_theme(self.draft)
        for role, (swatch, hex_label) in self.swatches.items():
            swatch.configure(bg=preview[role])
            hex_label.configure(text=str(preview[role]).upper())

    def _pick_color(self, role):
        current = build_theme(self.draft)[role]
        chosen = colorchooser.askcolor(color=current, parent=self.root, title="Choose a color")[1]
        if chosen:
            self.draft["colors"][role] = chosen
            self._refresh_settings_preview()

    def _reset_colors(self):
        self.draft["colors"] = {}
        self._refresh_settings_preview()

    def _read_draft_controls(self):
        self.draft["theme"] = self.theme_choices.get(self.theme_combo.get(), self.draft["theme"])
        self.draft["text_size"] = dict(TEXT_SIZES).get(self.size_combo.get(), 1.0)
        self.draft["font"] = self.font_combo.get().strip() or DEFAULT_SETTINGS["font"]
        self.draft["popup_text_size"] = dict(TEXT_SIZES).get(self.popup_size_combo.get(), 1.0)
        self.draft["popup_on_top"] = bool(self.on_top_var.get())
        self.draft["popup_show_customer"] = bool(self.show_customer_var.get())
        self.draft["popup_show_ask"] = bool(self.show_ask_var.get())
        self.draft["sounds_on"] = bool(self.sounds_var.get())
        self.draft["sound_volume"] = int(self.volume_scale.get())

    def apply_draft(self):
        self._read_draft_controls()
        self.apply_settings(self.draft)

    def apply_settings(self, new_settings):
        """Save the settings, switch to them and rebuild the windows, keeping the session."""
        global THEME, UI_FONT
        cleaned = _clean_settings(new_settings)
        try:
            save_settings(cleaned)
        except OSError as exc:
            messagebox.showerror("Could not save settings", str(exc), parent=self.root)
            return
        SETTINGS.clear()
        SETTINGS.update(cleaned)
        THEME = build_theme(SETTINGS, _ENV_THEME or None)
        UI_FONT = os.environ.get("MRAI_FONT", "").strip() or SETTINGS["font"]
        self._rebuild_ui()
        self.settings_status.configure(text="Applied. Your transcript, highlights and any open pop-up were kept.")

    def save_as_theme(self):
        name = self.theme_name_var.get().strip()
        if not name:
            self.settings_status.configure(text="Type a name for the theme first.")
            return
        if name.lower() in THEMES:
            self.settings_status.configure(text="That name belongs to a built-in theme. Choose another name.")
            return
        self._read_draft_controls()
        colors = {role: build_theme(self.draft)[role] for role in COLOR_ROLE_KEYS}
        self.draft["saved_themes"][name] = colors
        self.draft["theme"] = name
        self.draft["colors"] = {}
        self.apply_settings(self.draft)
        self.settings_status.configure(text=f'Saved as the theme "{name}" and applied.')

    def export_settings(self):
        path = filedialog.asksaveasfilename(parent=self.root, title="Export settings", defaultextension=".json",
                                            initialfile="rebuttal_butler_settings.json",
                                            filetypes=[("Settings file", "*.json")])
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as file:
                json.dump(_clean_settings(SETTINGS), file, indent=2)
        except OSError as exc:
            messagebox.showerror("Could not export", str(exc), parent=self.root)
            return
        self.settings_status.configure(text=f"Settings exported to {path}")

    def import_settings(self):
        path = filedialog.askopenfilename(parent=self.root, title="Import settings",
                                          filetypes=[("Settings file", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as file:
                raw = json.load(file)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not read that file", str(exc), parent=self.root)
            return
        if not isinstance(raw, dict):
            messagebox.showerror("Not a settings file", "That file does not contain Rebuttal Butler settings.",
                                 parent=self.root)
            return
        self.apply_settings(raw)
        self.settings_status.configure(text="Settings imported and applied.")

    # -- Script tab -----------------------------------------------------------------
    def _build_script_tab(self, parent):
        T = THEME
        if not hasattr(self, "script_sections"):
            self.script_sections = load_script_sections()
            self.script_index = 0 if self.script_sections else None
        card, inner = make_card(parent, None, pad=18)
        card.pack(fill="both", expand=True)
        tk.Label(inner, text="Call Script", font=font(15, "bold"), bg=T["surface"], fg=T["text"]).pack(anchor="w")
        intro = field_label(inner, "Pick a section on the left to write its script. Add a section for each part of "
                                   "your call. Changes are saved as you move between sections.")
        intro.pack(fill="x", pady=(px(2), px(12)))
        wrap_to_width(intro)

        columns = tk.Frame(inner, bg=T["surface"])
        columns.pack(fill="both", expand=True)
        left = tk.Frame(columns, bg=T["surface"], width=px(250))
        left.pack(side="left", fill="y", padx=(0, px(16)))
        left.pack_propagate(False)
        field_label(left, "Sections").pack(anchor="w")
        list_frame = tk.Frame(left, bg=T["surface2"], highlightthickness=1, highlightbackground=T["border"])
        list_frame.pack(fill="both", expand=True, pady=(px(4), px(8)))
        self.script_listbox = tk.Listbox(
            list_frame, exportselection=False, font=font(10), bg=T["surface2"], fg=T["text"],
            selectbackground=T["accent"], selectforeground=T["on_accent"], activestyle="none",
            relief="flat", bd=0, highlightthickness=0)
        list_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.script_listbox.yview,
                                    style="Mod.Vertical.TScrollbar")
        self.script_listbox.configure(yscrollcommand=list_scroll.set)
        self.script_listbox.pack(side="left", fill="both", expand=True, padx=(px(4), 0), pady=px(4))
        list_scroll.pack(side="right", fill="y")
        self.script_listbox.bind("<<ListboxSelect>>", self._script_on_select)
        field_label(left, "New section name").pack(anchor="w")
        self.script_new_var = tk.StringVar()
        make_entry(left, self.script_new_var).pack(fill="x", pady=(px(2), 0), ipady=px(5))
        FlatButton(left, "Add Section", self._script_add, padx=14).pack(anchor="w", pady=(px(7), 0))

        right = tk.Frame(columns, bg=T["surface"])
        right.pack(side="left", fill="both", expand=True)
        field_label(right, "Section name").pack(anchor="w")
        self.script_name_var = tk.StringVar()
        name_entry = make_entry(right, self.script_name_var)
        name_entry.pack(fill="x", pady=(px(2), px(8)), ipady=px(5))
        name_entry.bind("<FocusOut>", lambda e: self._script_commit())
        # Bottom items first, so a short window shrinks the script box and never hides the buttons.
        buttons = tk.Frame(right, bg=T["surface"])
        buttons.pack(side="bottom", fill="x", pady=(px(8), 0))
        FlatButton(buttons, "Save", self._script_commit, kind="primary", padx=14).pack(side="left")
        FlatButton(buttons, "Delete Section", self._script_delete, kind="danger", padx=14).pack(side="left", padx=(px(7), 0))
        self.script_status = field_label(buttons, "")
        self.script_status.pack(side="left", padx=px(12))
        field_label(right, "Script").pack(anchor="w")
        self.script_text = make_text(right, 12, size=11)
        self.script_text.pack(fill="both", expand=True, pady=(px(2), 0))
        self.script_text.bind("<FocusOut>", lambda e: self._script_sync())
        self._script_refresh_list()
        self._script_load_selected()

    def _script_has_selection(self):
        return self.script_index is not None and 0 <= self.script_index < len(self.script_sections)

    def _script_refresh_list(self):
        self.script_listbox.delete(0, "end")
        for section in self.script_sections:
            self.script_listbox.insert("end", section["name"])
        if self._script_has_selection():
            self.script_listbox.selection_clear(0, "end")
            self.script_listbox.selection_set(self.script_index)
            self.script_listbox.see(self.script_index)

    def _script_load_selected(self):
        text = self.script_sections[self.script_index]["text"] if self._script_has_selection() else ""
        name = self.script_sections[self.script_index]["name"] if self._script_has_selection() else ""
        self.script_name_var.set(name)
        self.script_text.delete("1.0", "end")
        if text:
            self.script_text.insert("1.0", text)

    def _script_sync(self):
        """Copy what is on screen into the section being edited. Does not save the file."""
        if not self._script_has_selection():
            return
        try:
            name = self.script_name_var.get().strip()
            text = self.script_text.get("1.0", "end-1c")
        except tk.TclError:
            return                                   # the window is being rebuilt
        section = self.script_sections[self.script_index]
        if name:
            section["name"] = name
        section["text"] = text

    def _script_write(self, message):
        try:
            save_script_sections(self.script_sections)
            self.script_status.configure(text=message)
        except (OSError, tk.TclError) as exc:
            self.script_status.configure(text=f"Could not save: {exc}")

    def _script_commit(self):
        """Save the section on screen and refresh the list, so a renamed section shows its new name."""
        self._script_sync()
        self._script_refresh_list()
        self._script_write("Saved.")

    def _script_on_select(self, _event=None):
        picked = self.script_listbox.curselection()
        if not picked or picked[0] == self.script_index:
            return
        self._script_sync()
        self._script_write("Saved.")
        self.script_index = picked[0]
        self._script_load_selected()
        self._script_refresh_list()

    def _script_add(self):
        name = self.script_new_var.get().strip()
        if not name:
            messagebox.showwarning("Name the section", "Type a name for the new section first.", parent=self.root)
            return
        self._script_sync()
        self.script_sections.append({"name": name, "text": ""})
        self.script_index = len(self.script_sections) - 1
        self.script_new_var.set("")
        self._script_load_selected()
        self._script_refresh_list()
        self._script_write("Section added.")

    def _script_delete(self):
        if not self._script_has_selection():
            return
        name = self.script_sections[self.script_index]["name"]
        if not messagebox.askyesno("Delete section", f"Delete the section \"{name}\" and its script?", parent=self.root):
            return
        del self.script_sections[self.script_index]
        self.script_index = min(self.script_index, len(self.script_sections) - 1) if self.script_sections else None
        self._script_load_selected()
        self._script_refresh_list()
        self._script_write("Section deleted.")

    def _build_library_tab(self, parent):
        T = THEME
        card, inner = make_card(parent, None, pad=18)
        card.pack(fill="both", expand=True)
        tk.Label(inner, text="Rebuttal Library", font=font(15, "bold"), bg=T["surface"], fg=T["text"]).pack(anchor="w")
        intro = field_label(inner, "Select an entry to edit its title, category, alternate customer phrasings, approved "
                                   "script, and follow-up. Click Save Changes to update live matching and save to disk.")
        intro.pack(fill="x", pady=(px(2), px(12)))
        wrap_to_width(intro)

        columns = tk.Frame(inner, bg=T["surface"])
        columns.pack(fill="both", expand=True)
        left = tk.Frame(columns, bg=T["surface"], width=px(250))
        left.pack(side="left", fill="y", padx=(0, px(16)))
        left.pack_propagate(False)
        field_label(left, "Saved objections").pack(anchor="w")
        list_frame = tk.Frame(left, bg=T["surface2"], highlightthickness=1, highlightbackground=T["border"])
        list_frame.pack(fill="both", expand=True, pady=(px(4), 0))
        self.library_listbox = tk.Listbox(
            list_frame, exportselection=False, font=font(10), bg=T["surface2"], fg=T["text"],
            selectbackground=T["accent"], selectforeground=T["on_accent"], activestyle="none",
            relief="flat", bd=0, highlightthickness=0)
        list_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.library_listbox.yview,
                                    style="Mod.Vertical.TScrollbar")
        self.library_listbox.configure(yscrollcommand=list_scroll.set)
        self.library_listbox.pack(side="left", fill="both", expand=True, padx=(px(4), 0), pady=px(4))
        list_scroll.pack(side="right", fill="y")
        self.library_listbox.bind("<<ListboxSelect>>", self.load_selected_library_item)

        right = tk.Frame(columns, bg=T["surface"])
        right.pack(side="left", fill="both", expand=True)
        field_label(right, "Objection title").pack(anchor="w")
        self.library_title_var = tk.StringVar()
        make_entry(right, self.library_title_var).pack(fill="x", pady=(px(2), px(8)), ipady=px(5))
        field_label(right, "Category").pack(anchor="w")
        self.library_category_var = tk.StringVar()
        make_entry(right, self.library_category_var).pack(fill="x", pady=(px(2), px(8)), ipady=px(5))
        field_label(right, "Alternate ways customers may say it (one per line)").pack(anchor="w")
        self.library_triggers_text = make_text(right, 5)
        self.library_triggers_text.pack(fill="x", pady=(px(2), px(8)))
        field_label(right, "Approved rebuttal script (exact wording)").pack(anchor="w")
        # Bottom items first, so a short window shrinks the script box and never hides the buttons.
        buttons = tk.Frame(right, bg=T["surface"])
        buttons.pack(side="bottom", fill="x")
        FlatButton(buttons, "Save Changes", self.save_library_item, kind="primary", padx=14).pack(side="left")
        FlatButton(buttons, "Add New", self.new_library_item, padx=14).pack(side="left", padx=(px(7), 0))
        FlatButton(buttons, "Delete Selected", self.delete_library_item, kind="danger", padx=14).pack(side="left", padx=(px(7), 0))
        FlatButton(buttons, "Cancel Edits", self.cancel_library_edits, padx=14).pack(side="left", padx=(px(7), 0))
        self.library_followup_text = make_text(right, 2)
        self.library_followup_text.pack(side="bottom", fill="x", pady=(px(2), px(10)))
        field_label(right, "Follow-up question").pack(side="bottom", anchor="w")
        self.library_response_text = make_text(right, 5)
        self.library_response_text.pack(fill="both", expand=True, pady=(px(2), px(8)))
        self.library_selected_index = None
        self.library_is_new = False
        self.refresh_library_list()
        if OBJECTIONS:
            self.library_listbox.selection_set(0)
            self.load_selected_library_item()

    # -- Library tab -------------------------------------------------------------
    def refresh_library_list(self, select_index=None):
        self.library_listbox.delete(0, "end")
        for item in OBJECTIONS:
            self.library_listbox.insert("end", item.get("title", "Untitled objection"))
        if select_index is not None and 0 <= select_index < len(OBJECTIONS):
            self.library_listbox.selection_clear(0, "end")
            self.library_listbox.selection_set(select_index)
            self.library_listbox.see(select_index)
            self.library_selected_index = select_index

    def load_selected_library_item(self, _event=None):
        selected = self.library_listbox.curselection()
        if not selected:
            return
        index = selected[0]
        if not self.library_is_new:
            self.library_selected_index = index
        item = OBJECTIONS[index]
        self.library_is_new = False
        self.library_title_var.set(item.get("title", ""))
        self.library_category_var.set(item.get("category", ""))
        self._replace_text(self.library_triggers_text, "\n".join(item.get("triggers", [])))
        self._replace_text(self.library_response_text, item.get("response", ""))
        self._replace_text(self.library_followup_text, item.get("followup", ""))

    @staticmethod
    def _replace_text(widget, value):
        widget.delete("1.0", "end")
        widget.insert("1.0", value or "")

    def save_library_item(self):
        title = self.library_title_var.get().strip()
        category = self.library_category_var.get().strip() or "General"
        triggers = [line.strip() for line in self.library_triggers_text.get("1.0", "end").splitlines() if line.strip()]
        response = self.library_response_text.get("1.0", "end").strip()
        followup = self.library_followup_text.get("1.0", "end").strip()
        if not title or not response:
            messagebox.showwarning("Missing information", "An objection title and approved rebuttal script are required.", parent=self.root)
            return
        triggers = list(dict.fromkeys([title] + triggers))
        for i, existing in enumerate(OBJECTIONS):
            if i != self.library_selected_index and normalize(existing.get("title", "")) == normalize(title):
                messagebox.showwarning("Duplicate title", "Another library entry already uses that title.", parent=self.root)
                return
        new_item = {"title": title, "category": category, "triggers": triggers, "response": response, "followup": followup}
        if self.library_is_new or self.library_selected_index is None:
            OBJECTIONS.append(new_item)
            target_index = len(OBJECTIONS) - 1
        else:
            target_index = self.library_selected_index
            OBJECTIONS[target_index].clear()
            OBJECTIONS[target_index].update(new_item)
        USER_ADDED_OBJECTIONS[:] = [item for item in OBJECTIONS if normalize(item.get("category", "")) == "user added"]
        try:
            save_user_library()
        except OSError as exc:
            messagebox.showerror("Save failed", f"The library could not be saved:\n{exc}", parent=self.root)
            return
        self.library_is_new = False
        self.refresh_library_list(target_index)
        self.status_var.set(f"Saved library entry: {title}")
        messagebox.showinfo("Saved", f'"{title}" was saved and is available for live matching.', parent=self.root)

    def new_library_item(self):
        self.library_listbox.selection_clear(0, "end")
        self.library_selected_index = None
        self.library_is_new = True
        self.library_title_var.set("")
        self.library_category_var.set("User added")
        self._replace_text(self.library_triggers_text, "")
        self._replace_text(self.library_response_text, "")
        self._replace_text(self.library_followup_text, "")

    def delete_library_item(self):
        if self.library_selected_index is None or self.library_is_new:
            messagebox.showinfo("Nothing selected", "Select a saved objection first.", parent=self.root)
            return
        item = OBJECTIONS[self.library_selected_index]
        if not messagebox.askyesno("Delete rebuttal", f'Delete "{item.get("title", "Untitled")}" from the library? This cannot be undone.', parent=self.root):
            return
        del OBJECTIONS[self.library_selected_index]
        USER_ADDED_OBJECTIONS[:] = [entry for entry in OBJECTIONS if normalize(entry.get("category", "")) == "user added"]
        try:
            save_user_library()
        except OSError as exc:
            messagebox.showerror("Save failed", f"The library could not be saved:\n{exc}", parent=self.root)
            return
        self.library_selected_index = None
        self.library_is_new = False
        self.refresh_library_list(0 if OBJECTIONS else None)
        if OBJECTIONS:
            self.load_selected_library_item()
        else:
            self.new_library_item()

    def cancel_library_edits(self):
        if self.library_is_new:
            if OBJECTIONS:
                self.library_is_new = False
                self.refresh_library_list(0 if self.library_selected_index is None else self.library_selected_index)
                self.load_selected_library_item()
            else:
                self.new_library_item()
            return
        if self.library_selected_index is not None and 0 <= self.library_selected_index < len(OBJECTIONS):
            self.load_selected_library_item()

    def accept_new_rebuttal(self):
        title = self.new_title_var.get().strip()
        response = self.new_response_text.get("1.0", "end").strip()
        triggers = [line.strip() for line in self.new_triggers_text.get("1.0", "end").splitlines() if line.strip()]
        followup = self.new_followup_var.get().strip()
        if not title:
            messagebox.showwarning("Missing objection", "Enter a short objection title first.", parent=self.root)
            return
        if not response:
            messagebox.showwarning("Missing rebuttal", "Paste the approved rebuttal script first.", parent=self.root)
            return
        if not triggers:
            triggers = [title]
        if len(triggers) < 10:
            answer = messagebox.askyesno("Fewer than 10 variations", f"You entered {len(triggers)} alternate phrasings. Add it anyway? You can edit the library file later to add more.", parent=self.root)
            if not answer:
                return
        if any(normalize(item.get("title", "")) == normalize(title) for item in OBJECTIONS):
            messagebox.showwarning("Already exists", "An objection with this title is already in the library. Use a distinct title or edit the existing library file.", parent=self.root)
            return
        item = {"title": title, "category": "User added", "triggers": list(dict.fromkeys([title] + triggers)), "response": response, "followup": followup}
        OBJECTIONS.append(item)
        USER_ADDED_OBJECTIONS.append(item)
        try:
            save_user_library()
        except OSError as exc:
            OBJECTIONS.remove(item)
            USER_ADDED_OBJECTIONS.remove(item)
            messagebox.showerror("Could not save", str(exc), parent=self.root)
            return
        self.status_var.set(f"Added rebuttal: {title}")
        messagebox.showinfo("Rebuttal added", f'"{title}" was added to the library and is ready for matching.', parent=self.root)
        self.clear_rebuttal_form()
        self.refresh_library_list()

    def clear_rebuttal_form(self):
        self.new_title_var.set("")
        self.new_triggers_text.delete("1.0", "end")
        self.new_response_text.delete("1.0", "end")
        self.new_followup_var.set("")

    # -- Suggested-response window -----------------------------------------------
    def _build_response_window(self):
        global _FONT_SCALE_OVERRIDE
        _FONT_SCALE_OVERRIDE = THEME["scale"] * float(SETTINGS.get("popup_text_size", 1.0))
        try:
            self._build_response_window_body()
        finally:
            _FONT_SCALE_OVERRIDE = None
        self.response_window.bind("<Configure>", self._on_tour_configure, add="+")

    def _build_response_window_body(self):
        T = THEME
        win = self.response_window = tk.Toplevel(self.root)
        win.title("Suggested Rebuttal")
        set_window_icon(win)
        grow = _FONT_SCALE_OVERRIDE or T["scale"]
        screen_w, screen_h = win.winfo_screenwidth(), win.winfo_screenheight()
        width = min(px(580 * grow), screen_w - px(40))
        height = min(px(700 * grow), screen_h - px(110))
        left = min(px(960), max(0, screen_w - width - px(10)))      # stay on screen
        win.geometry(f"{width}x{height}+{left}+{px(90)}")
        win.minsize(min(px(530 * grow), width), min(px(540 * grow), height))
        win.configure(bg=T["bg"])
        win.attributes("-topmost", bool(SETTINGS.get("popup_on_top", True)))
        win.protocol("WM_DELETE_WINDOW", self.hide_response_window)

        outer = tk.Frame(win, bg=T["bg"])
        outer.pack(fill="both", expand=True, padx=px(20), pady=px(18))
        self.tour_targets["popup"] = outer          # the tutorial glow sits in this padding

        top = tk.Frame(outer, bg=T["bg"])
        top.pack(fill="x")
        self.match_chip = tk.Label(top, text="READY", font=font(8, "bold"), padx=px(10), pady=px(3),
                                   bg=T["border"], fg=T["muted"])
        self.match_chip.pack(side="left")
        self._pop_logo = load_logo(px(84))
        if self._pop_logo is not None:
            tk.Label(top, image=self._pop_logo, bg=T["bg"], bd=0).pack(side="right", padx=(px(10), 0))
        self.match_counter = tk.Label(top, text="", font=font(9), bg=T["bg"], fg=T["muted"])
        self.match_counter.pack(side="right")

        self.match_title = tk.Label(outer, text="Waiting for an objection…", font=font(19, "bold"),
                                    bg=T["bg"], fg=T["text"], anchor="w", justify="left")
        self.match_title.pack(fill="x", pady=(px(12), px(2)))
        wrap_to_width(self.match_title)

        self.customer_label = tk.Label(outer, text="CUSTOMER SAID", font=font(8, "bold"), bg=T["bg"],
                                       fg=T["muted"])
        self.customer_quote = make_text(outer, 3, size=11, bg_key="surface", style="italic")
        if SETTINGS.get("popup_show_customer", True):
            self.customer_label.pack(anchor="w", pady=(px(12), px(4)))
            self.customer_quote.pack(fill="x")
        self.customer_quote.configure(state="disabled")

        say = tk.Frame(outer, bg=T["surface"], highlightthickness=1, highlightbackground=T["border"])
        say.pack(fill="both", expand=True, pady=(px(14), 0))
        tk.Frame(say, bg=T["accent"], width=px(4)).pack(side="left", fill="y")
        say_inner = tk.Frame(say, bg=T["surface"])
        say_inner.pack(side="left", fill="both", expand=True)
        tk.Label(say_inner, text="SAY THIS", font=font(8, "bold"), bg=T["surface"],
                 fg=T["accent"]).pack(anchor="w", padx=px(14), pady=(px(12), 0))
        self.response_text = make_text(say_inner, 5, size=13, bg_key="surface", border=False)
        self.response_text.pack(fill="both", expand=True, padx=(px(4), px(8)), pady=(px(2), px(8)))
        self.response_text.configure(state="disabled")

        ask = tk.Frame(outer, bg=T["surface"], highlightthickness=1, highlightbackground=T["border"])
        ask.pack(fill="x", pady=(px(12), 0))
        tk.Frame(ask, bg=T["warn"], width=px(4)).pack(side="left", fill="y")
        ask_inner = tk.Frame(ask, bg=T["surface"])
        ask_inner.pack(side="left", fill="x", expand=True)
        tk.Label(ask_inner, text="ASK NEXT", font=font(8, "bold"), bg=T["surface"],
                 fg=T["warn"]).pack(anchor="w", padx=px(14), pady=(px(10), 0))
        self.followup_label = tk.Label(ask_inner, text="", font=font(11), bg=T["surface"], fg=T["text"],
                                       anchor="w", justify="left")
        self.followup_label.pack(fill="x", padx=px(14), pady=(px(2), px(10)))
        wrap_to_width(self.followup_label, margin=30)

        buttons = tk.Frame(outer, bg=T["bg"])
        buttons.pack(fill="x", pady=(px(14), 0))
        FlatButton(buttons, "Good Match", self.mark_good_match, kind="primary", padx=14).pack(side="left")
        self.next_button = FlatButton(buttons, "Other Suggestion", self.next_suggestion, padx=14)
        self.next_button.pack(side="left", padx=px(6))
        self.copy_button = FlatButton(buttons, "Copy Response", self.copy_response, padx=14)
        self.copy_button.pack(side="left")
        FlatButton(buttons, "Hide", self.hide_response_window, padx=14).pack(side="right")
        # Pack order decides who keeps their space when the window is short: the buttons
        # first, then the follow-up question, then the script card takes what's left.
        buttons.pack_forget()
        ask.pack_forget()
        say.pack_forget()
        buttons.pack(side="bottom", fill="x", pady=(px(14), 0))
        if SETTINGS.get("popup_show_ask", True):
            ask.pack(side="bottom", fill="x", pady=(px(12), 0))
        say.pack(fill="both", expand=True, pady=(px(14), 0))
        self._show_waiting()

    # -- Audio devices -------------------------------------------------------------
    def refresh_devices(self):
        labels = []
        self.sources = {}
        default_name, outputs = desktop_audio_info()
        if pyaudio is not None:
            label = "Desktop audio: this PC's speakers" + (f" ({default_name})" if default_name else "") + " (recommended)"
            labels.append(label)
            self.sources[label] = ("desktop", None)
            for index, name in outputs:
                if name != default_name:
                    label = f"Desktop audio: {name}"
                    labels.append(label)
                    self.sources[label] = ("desktop", index)
        input_error = None
        if sd is not None:
            try:
                devices = sd.query_devices()
                try:
                    hostapis = sd.query_hostapis()
                except Exception:
                    hostapis = []
                for index, device in enumerate(devices):
                    if int(device.get("max_input_channels", 0)) > 0:
                        try:
                            api = f" [{hostapis[device['hostapi']]['name']}]"
                        except Exception:
                            api = ""
                        label = f"Input: {index}: {device['name']}{api}"
                        labels.append(label)
                        self.sources[label] = ("input", index)
            except Exception as exc:
                input_error = exc
        if not labels and input_error is not None:
            self.device_combo["values"] = []
            self.device_var.set("")
            self.status_var.set(f"Could not list audio devices: {input_error}")
            messagebox.showerror("Audio device error", str(input_error), parent=self.root)
            return
        mics = [label for label in labels if label.startswith("Input: ")]
        self.mine_sources = {label: self.sources[label] for label in mics}
        choices = ["None (don't transcribe my side)"] + mics
        self.mine_combo["values"] = choices
        if self.mine_var.get() in choices:
            self.mine_combo.set(self.mine_var.get())
        else:
            preferred = next((label for label in mics if "microphone" in label.lower()), None)
            self.mine_combo.set(preferred or (mics[0] if mics else choices[0]))
        self.device_combo["values"] = labels
        if labels:
            current = self.device_var.get()
            if current in labels:
                self.device_combo.set(current)
            else:
                self.device_combo.current(0)
            self.status_var.set(f"Found {len(labels)} audio source(s).")
        else:
            self.device_var.set("")
            self.status_var.set("No audio sources found.")

    def _selected_source(self):
        return self.sources.get(self.device_var.get())

    def test_audio(self):
        """Listen to the chosen source for 3 seconds and say what came through."""
        if self.is_listening or self.engine.busy:
            self.status_var.set("Stop listening first, then test the audio.")
            return
        if self.testing_audio:
            return
        source = self._selected_source()
        if source is None:
            messagebox.showwarning("Choose an audio source", "Select what to listen to first.", parent=self.root)
            return
        self.testing_audio = True
        self.test_button.configure(state="disabled")
        self.status_var.set("Testing for 3 seconds… play the call (or any sound) now.")
        threading.Thread(target=self._test_audio_worker, args=(source,), daemon=True).start()

    def _test_audio_worker(self, source):
        try:
            blocks, peak = self.engine.probe(source, 3.0)
            if blocks == 0 and source[0] == "desktop":
                message = ("Connected, but Windows sent no audio. That is normal while the PC is silent: "
                           "play the call (or any sound) through the selected speakers and test again.")
            elif blocks == 0:
                message = "Connected, but this input sent no audio."
            elif peak < 0.002:
                message = "Connected and receiving, but it is silent right now. Play the call (or any sound) and test again."
            else:
                level = int(min(100.0, math.sqrt(peak) / 0.5 * 100.0))
                message = f"Hearing audio (level {level}%). You can click Start Listening."
        except Exception as exc:
            message = f"Test failed: {exc}"
        self.events.put(("tested", message))

    # -- Start / stop ------------------------------------------------------------------
    def _selected_mine_source(self):
        return self.mine_sources.get(self.mine_var.get())

    def _record_side(self):
        """'mine', 'caller' or 'both', from the Record dropdown."""
        return dict(RECORD_CHOICES).get(self.record_var.get(), "both")

    def _records_caller(self):
        return self._record_side() in ("caller", "both")

    def _records_mine(self):
        return self._record_side() in ("mine", "both")

    def _new_recording_path(self, label):
        """A new file name in audiofiles. It never overwrites an earlier recording."""
        folder = self._audio_folder()
        stamp = time.strftime("%Y-%m-%d %H-%M-%S")
        path = os.path.join(folder, f"{stamp} {label}.mp3")
        number = 2
        while os.path.exists(path):
            path = os.path.join(folder, f"{stamp} {label} {number}.mp3")
            number += 1
        return path

    def _refresh_record_dot(self):
        self.record_dot.set_state(armed=self.recording_armed, blinking=self.recording_now)
        self.record_combo.configure(state="disabled" if self.recording_now else "readonly")

    def _apply_gains(self):
        self.engine.gain = 10 ** (self.gain_db["caller"] / 20.0)
        self.mine_engine.gain = 10 ** (self.gain_db["mine"] / 20.0)

    def _open_call_file(self):
        """Open the one MP3 for this recording and connect the chosen sides to it."""
        records_caller, records_mine = self._records_caller(), self._records_mine()
        channels = 2 if (records_caller and records_mine) else 1
        self.call_file = CallRecorder(self._new_recording_path("call"), channels)
        self.engine.sink = self.call_file if records_caller else None
        self.engine.channel = 0
        self.mine_engine.sink = self.call_file if records_mine else None
        self.mine_engine.channel = 1

    def _close_call_file(self):
        """Stop both sides feeding the file, finish the MP3, and return its path (or None)."""
        self.engine.sink = None
        self.mine_engine.sink = None
        call, self.call_file = self.call_file, None
        if call is None:
            return None
        call.close()
        return call.path

    def _finish_recordings(self):
        """Close the recording file. Returns the name of the file that was saved, in a list."""
        was_recording = self.recording_now
        self.recording_now = False
        saved = []
        try:
            path = self._close_call_file()
        except Exception as exc:
            self.status_var.set(f"Could not finish the recording: {exc}")
            path = None
        if path:
            saved.append(os.path.basename(path))
            self._recording_saved(path)
        if was_recording:
            play_sound("record_stop")
        self._refresh_record_dot()
        self.refresh_recording_list()
        return saved

    def toggle_recording(self):
        """The round button. Turns recording on or off, during a call too."""
        if self.recording_armed:
            self.recording_armed = False
            if self.recording_now:
                saved = self._finish_recordings()
                self.status_var.set("Recording stopped." + (f"  Saved: {', '.join(saved)}" if saved else ""))
            else:
                self.status_var.set("Recording is off.")
        elif self._records_mine() and self._selected_mine_source() is None:
            messagebox.showwarning("Choose your microphone",
                                   "To record your side of the call, pick your microphone under "
                                   "My microphone first.", parent=self.root)
        else:
            self.recording_armed = True
            if self.is_listening:
                try:
                    self._apply_gains()
                    self._open_call_file()
                except Exception as exc:
                    self.recording_armed = False
                    self.status_var.set(f"Could not start recording: {exc}")
                    self._refresh_record_dot()
                    return
                self.recording_now = True
                play_sound("record_start")
                self.status_var.set("Recording… listening carries on.")
            else:
                self.status_var.set("Recording is on. It starts when you click Start Listening.")
        self._refresh_record_dot()

    def start_listening(self):
        if self.is_listening:
            return
        if self.engine.busy or self.mine_engine.busy:
            self.status_var.set("Still shutting down the previous session. Try again in a moment.")
            return
        source = self._selected_source()
        if source is None:
            messagebox.showwarning("Choose an audio source", "Select what to listen to first.", parent=self.root)
            return
        if self.testing_audio:
            self.status_var.set("Wait for the audio test to finish, then click Start Listening.")
            return
        mine_source = self._selected_mine_source()
        if self.recording_armed and self._records_mine() and mine_source is None:
            messagebox.showwarning("Choose your microphone",
                                   "To record your side of the call, pick your microphone under "
                                   "My microphone first.", parent=self.root)
            return
        self._apply_gains()
        if self.recording_armed:
            try:
                self._open_call_file()
            except Exception as exc:
                messagebox.showerror("Could not record", f"The recording file could not be opened.\n\n{exc}",
                                     parent=self.root)
                return
        self.listening_source_kind = source[0]
        self.last_final_utt_id = 0
        self.last_partial_text = ""
        self.current_live_match_title = ""
        self.listening_live = False
        self.no_audio_warned = False
        self.is_listening = True
        self.recording_now = self.recording_armed
        self.saved_names = []
        self.recording_notice = ""
        self.pending_mine_start = mine_source
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set("Preparing speech recognition…")
        self._set_preview("Starting live transcription…")
        self._set_state("loading")
        self._refresh_record_dot()
        if self.recording_now:
            play_sound("record_start")
        self.engine.start(source)

    def stop_listening(self):
        if not self.is_listening and not self.engine.busy and not self.mine_engine.busy:
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            return
        self.is_listening = False
        self.listening_live = False
        self.pending_mine_start = None
        # Start stays disabled until both sides have fully stopped, so a second transcription
        # worker can never overlap an unfinished one.
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.status_var.set("Stopping…")
        if self.ui_state != "error":
            self._set_state("idle")
        self._finish_recordings()
        self.mine_engine.stop()
        self.engine.stop()

    def _listener_finished(self):
        self.is_listening = False
        self.listening_live = False
        self.recording_now = False
        self._refresh_record_dot()
        self.start_button.configure(state="disabled" if self.mine_engine.busy else "normal")
        self.stop_button.configure(state="disabled")
        self.level_bar.configure(value=0)
        if not self.status_var.get().startswith("Listener error"):
            notice = f"  {self.recording_notice}" if self.recording_notice else ""
            self.status_var.set("Listening stopped." + notice)
        if self.ui_state != "error":
            self._set_state("ready" if self.engine.model is not None else "idle")

    def _start_my_side(self):
        """Start the microphone once the caller side is listening."""
        mine_source, self.pending_mine_start = self.pending_mine_start, None
        if mine_source is None or not self.is_listening:
            return
        self.mine_engine.start(mine_source)

    def _recording_saved(self, path):
        self.saved_names.append(os.path.basename(path))
        self.recording_notice = "Saved to audiofiles: " + ", ".join(self.saved_names)
        if not self.is_listening:
            self.status_var.set(self.recording_notice)

    def _update_rec_meter(self, key, rms):
        meter = self.rec_meters.get(key)
        if meter is None:
            return
        try:
            meter.configure(value=min(100.0, math.sqrt(max(0.0, rms)) / 0.5 * 100.0))
        except tk.TclError:
            pass

    def _handle_mine_event(self, event):
        kind = event[0]
        if kind == "rec_level":
            self._update_rec_meter("mine", event[1])
        elif kind == "error":
            self.status_var.set(f"My microphone could not be opened: {event[1]}")
        elif kind == "status" and self.is_listening and not event[1].startswith("Loading"):
            self.status_var.set(f"My microphone: {event[1]}")
        elif kind == "finished" and not self.is_listening and not self.engine.busy:
            self.start_button.configure(state="normal")

    def _set_gain(self, key, value):
        db = float(value)
        self.gain_db[key] = db
        engine = self.engine if key == "caller" else self.mine_engine
        engine.gain = 10 ** (db / 20.0)
        scale = self.gain_scales.get(key)
        if scale is not None:
            try:
                scale[1].configure(text=self._gain_text(db))
            except tk.TclError:
                pass

    @staticmethod
    def _gain_text(db):
        return f"{db:+.0f} dB"

    def _set_playback_volume(self, value):
        self.playback_volume = float(value)

    def _build_recordings_tab(self, parent):
        T = THEME
        card, inner = make_card(parent, None, pad=18)
        card.pack(fill="x", pady=(0, px(10)))
        tk.Label(inner, text="Levels and gain", font=font(15, "bold"), bg=T["surface"], fg=T["text"]).pack(anchor="w")
        intro = field_label(inner, "Gain makes what is recorded louder or quieter, for a quiet caller or headset. "
                                   "It does not change what the transcript hears. The bars show the level being "
                                   "recorded. Changes apply right away, including during a call.")
        intro.pack(fill="x", pady=(px(2), px(10)))
        wrap_to_width(intro)
        self.rec_meters = {}
        self.gain_scales = {}
        for key, label in (("caller", "Caller (desktop audio)"), ("mine", "My microphone")):
            row = tk.Frame(inner, bg=T["surface"])
            row.pack(fill="x", pady=px(4))
            tk.Label(row, text=label, font=font(10), bg=T["surface"], fg=T["text"], width=22,
                     anchor="w").pack(side="left")
            meter = LevelMeter(row)
            meter.pack(side="left", fill="x", expand=True, padx=(0, px(10)))
            self.rec_meters[key] = meter
            readout = tk.Label(row, text=self._gain_text(self.gain_db[key]), font=font(9), bg=T["surface"],
                               fg=T["muted"], width=8)
            scale = tk.Scale(row, from_=-12, to=18, orient="horizontal", length=px(200), resolution=1,
                             showvalue=False, bg=T["surface"], fg=T["text"], troughcolor=T["surface2"],
                             highlightthickness=0, bd=0, activebackground=T["accent"],
                             command=lambda value, k=key: self._set_gain(k, value))
            scale.set(int(self.gain_db[key]))
            scale.pack(side="right")
            readout.pack(side="right", padx=(px(8), 0))
            tk.Label(row, text="Gain", font=font(9), bg=T["surface"], fg=T["muted"]).pack(side="right", padx=(px(8), 0))
            self.gain_scales[key] = (scale, readout)

        card, inner = make_card(parent, None, pad=18)
        card.pack(fill="both", expand=True)
        tk.Label(inner, text="Saved recordings", font=font(15, "bold"), bg=T["surface"], fg=T["text"]).pack(anchor="w")
        intro = field_label(inner, "Each call is one MP3. Both sides are recorded into both the left and right "
                                   "channels. Volume here changes playback only.")
        intro.pack(fill="x", pady=(px(2), px(10)))
        wrap_to_width(intro)
        buttons = tk.Frame(inner, bg=T["surface"])
        buttons.pack(fill="x", pady=(0, px(8)))
        FlatButton(buttons, "Play", self.play_selected_recording, kind="primary").pack(side="left")
        FlatButton(buttons, "Stop", self.stop_playback).pack(side="left", padx=(px(8), 0))
        FlatButton(buttons, "Refresh list", self.refresh_recording_list).pack(side="left", padx=(px(8), 0))
        FlatButton(buttons, "Open folder", self.open_recordings_folder).pack(side="left", padx=(px(8), 0))
        tk.Label(buttons, text="Playback volume", font=font(9), bg=T["surface"], fg=T["muted"]).pack(
            side="right", padx=(px(8), 0))
        volume = tk.Scale(buttons, from_=0, to=150, orient="horizontal", length=px(180), showvalue=False,
                          bg=T["surface"], fg=T["text"], troughcolor=T["surface2"], highlightthickness=0, bd=0,
                          activebackground=T["accent"], command=self._set_playback_volume)
        volume.set(int(self.playback_volume))
        volume.pack(side="right")
        self.recording_list = tk.Listbox(inner, height=8, bg=T["surface2"], fg=T["text"],
                                         selectbackground=T["select"], selectforeground=T["text"], relief="flat",
                                         bd=0, highlightthickness=1, highlightbackground=T["border"],
                                         font=font(10), activestyle="none")
        self.recording_list.pack(fill="both", expand=True)
        self.recording_list.bind("<Double-Button-1>", lambda _e: self.play_selected_recording())
        self.recording_status = tk.Label(inner, text="", font=font(9), bg=T["surface"], fg=T["muted"], anchor="w")
        self.recording_status.pack(fill="x", pady=(px(6), 0))
        self.refresh_recording_list()

    def refresh_recording_list(self):
        if not hasattr(self, "recording_list"):
            return
        try:
            folder = self._audio_folder()
        except OSError:
            folder = None
        files = []
        if folder and os.path.isdir(folder):
            files = sorted((name for name in os.listdir(folder) if name.lower().endswith(".mp3")), reverse=True)
        self.recording_paths = [os.path.join(folder, name) for name in files]
        self.recording_list.delete(0, "end")
        for name in files:
            self.recording_list.insert("end", name)
        self.recording_status.configure(
            text=f"{len(files)} recording(s) in audiofiles." if files else "No recordings yet. They appear here after you stop one.")

    def play_selected_recording(self):
        picked = self.recording_list.curselection()
        if not picked or picked[0] >= len(self.recording_paths):
            self.recording_status.configure(text="Pick a recording in the list first.")
            return
        if miniaudio is None or sd is None:
            self.recording_status.configure(text="Playback is not available on this PC.")
            return
        path = self.recording_paths[picked[0]]
        try:
            decoded = miniaudio.mp3_read_file_f32(path)
        except Exception as exc:
            self.recording_status.configure(text=f"Could not open that recording: {exc}")
            return
        samples = np.asarray(decoded.samples, dtype=np.float32).reshape(-1, decoded.nchannels)
        self.stop_playback()
        self._play_data = samples
        self._play_pos = 0
        try:
            stream = sd.OutputStream(samplerate=decoded.sample_rate, channels=decoded.nchannels,
                                     dtype="float32", callback=self._play_callback)
            stream.start()
        except Exception as exc:
            self.recording_status.configure(text=f"Could not play it: {exc}")
            return
        self.playback = stream
        self.recording_status.configure(text=f"Playing {os.path.basename(path)}")
        if not self._watching:
            self._watching = True
            self.root.after(300, self._watch_playback)

    def _play_callback(self, outdata, frames, time_info, status):
        data = self._play_data
        start = self._play_pos
        chunk = data[start:start + frames]
        out = np.zeros((frames, outdata.shape[1]), dtype=np.float32)
        if len(chunk):
            out[:len(chunk), :chunk.shape[1]] = chunk * np.float32(self.playback_volume / 100.0)
        outdata[:] = np.clip(out, -1.0, 1.0)
        self._play_pos = start + len(chunk)
        if self._play_pos >= len(data):
            raise sd.CallbackStop()

    def _watch_playback(self):
        stream = self.playback
        if stream is None:
            self._watching = False
            return
        try:
            active = stream.active
        except Exception:
            active = False
        if active:
            self.root.after(300, self._watch_playback)
            return
        self._watching = False
        self.stop_playback()
        try:
            self.recording_status.configure(text="Finished playing.")
        except tk.TclError:
            pass

    def stop_playback(self):
        stream, self.playback = self.playback, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        try:
            if hasattr(self, "recording_status") and stream is not None:
                self.recording_status.configure(text="Stopped.")
        except tk.TclError:
            pass

    def open_recordings_folder(self):
        try:
            folder = self._audio_folder()
        except OSError as exc:
            self.recording_status.configure(text=f"Could not open the folder: {exc}")
            return
        try:
            os.startfile(folder)                    # Windows
        except AttributeError:
            self.recording_status.configure(text=f"Recordings are in: {folder}")
        except OSError as exc:
            self.recording_status.configure(text=f"Could not open the folder: {exc}")

    def _test_sound(self):
        play_sound("select", volume_percent=int(self.volume_scale.get()), force=True)

    def _worker_error(self, message, details):
        self.stop_listening()
        self.status_var.set(f"Listener error: {message}")
        self._set_state("error")
        messagebox.showerror("Listener error", f"{message}\n\nFull details:\n{details}", parent=self.root)

    # -- Events from the engine (always handled on the UI thread) ---------------------
    def _poll_events(self):
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            try:
                self._handle_event(event)
            except Exception:
                traceback.print_exc()
        while True:
            try:
                event = self.mine_events.get_nowait()
            except queue.Empty:
                break
            try:
                self._handle_mine_event(event)
            except Exception:
                traceback.print_exc()
        if (self.is_listening and self.listening_live and not self.no_audio_warned
                and time.time() - self.last_sound_time > 10):
            self.no_audio_warned = True
            if self.listening_source_kind == "desktop":
                self.status_var.set("No call audio for 10 seconds. Check that the call plays through the speakers or "
                                    "headset chosen above (if you switched headsets, click Stop and Start again).")
            else:
                self.status_var.set("No audio for 10 seconds. If a call is playing, check the selected input device and your sound routing.")
        self.root.after(50, self._poll_events)

    def _handle_event(self, event):
        kind = event[0]
        if kind == "status":
            self.status_var.set(event[1])
            if event[1].startswith("Loading"):
                self._set_state("loading")
        elif kind == "ready":
            if not self.is_listening:
                self.status_var.set(f"Ready: Whisper {event[1]}. Click Start Listening.")
                self._set_state("ready")
        elif kind == "listening":
            self.listening_live = True
            self.last_sound_time = time.time()
            self.no_audio_warned = False
            self.status_var.set(f"Listening… (Whisper {event[1]})" + ("  ·  Recording" if self.recording_now else ""))
            self._set_state("listening")
            self._start_my_side()
        elif kind == "saved":
            self._recording_saved(event[1])
        elif kind == "rec_level":
            self._update_rec_meter("caller", event[1])
        elif kind == "level":
            self._update_level(event[1])
        elif kind == "partial":
            self.handle_partial_text(event[1], event[2])
        elif kind == "final":
            _, utt_id, text, audio_seconds, took = event
            self.handle_final_text(utt_id, text)
            if self.is_listening:
                # Whisper's cost per phrase is nearly fixed, so the wait itself is the
                # number that matters in a live call, whatever the phrase length.
                note = "" if took <= 1.5 else "  (slow: use a smaller model or a GPU, see benchmark.py)"
                self.status_var.set(f"Listening… last phrase took {took:.1f}s to transcribe{note}")
        elif kind == "tested":
            self.testing_audio = False
            self.test_button.configure(state="normal")
            if not self.is_listening:
                self.status_var.set(event[1])
        elif kind == "error":
            self._worker_error(event[1], event[2])
        elif kind == "finished":
            self._listener_finished()

    def _update_level(self, rms):
        self.level_bar.configure(value=min(100.0, math.sqrt(rms) / 0.5 * 100.0))
        if rms > 0.0005:
            self.last_sound_time = time.time()
            if self.no_audio_warned:
                self.no_audio_warned = False
                self.status_var.set("Listening…")

    # -- Transcript handling -----------------------------------------------------------
    def handle_partial_text(self, utt_id, text):
        """Show replaceable interim words and surface a likely objection early."""
        if utt_id <= self.last_final_utt_id:
            return                                  # that phrase already has its final text
        cleaned = normalize(text)
        if not cleaned:
            return
        self.last_partial_text = text
        self._set_preview(text + " …")
        # Very short or unfinished snippets are noisy: only exact trigger phrases count here.
        if len(cleaned.split()) < 3:
            return
        matches = match_objections(text, strict=True)
        if not matches:
            return
        best_title = matches[0].get("title", "")
        if best_title != self.current_live_match_title:
            self.current_live_match_title = best_title
            self.show_response(matches, text, raise_window=True)
        elif self.current_matches:
            # Same suggestion: refresh the quote without stealing focus from the dialer.
            self.current_matches = matches
            self.current_customer_text = text
            self.display_current_match()

    def handle_final_text(self, utt_id, text):
        self.last_final_utt_id = max(self.last_final_utt_id, utt_id)
        cleaned = normalize(text)
        had_partial = bool(self.last_partial_text)
        self.last_partial_text = ""
        if not cleaned:
            if had_partial:
                self._set_preview("(no clear speech recognised)")
            self.current_live_match_title = ""
            return
        now = time.time()
        if cleaned == self.last_final_text and now - self.last_final_time < 8:
            return
        self.last_final_text = cleaned
        self.last_final_time = now
        self._set_preview(text)
        self.history_text.insert("end", time.strftime("%H:%M:%S"), "ts")
        self.history_text.insert("end", f"   {text}\n\n")
        self.history_text.see("end")
        matches = match_objections(text)
        if matches:
            changed = matches[0].get("title", "") != self.current_live_match_title
            self.show_response(matches, text, raise_window=changed)
        self.current_live_match_title = ""          # the next phrase starts fresh

    def _set_preview(self, text):
        self.preview_text.configure(state="normal")
        self.preview_text.delete("1.0", "end")
        self.preview_text.insert("1.0", text)
        self.preview_text.configure(state="disabled")

    def show_response(self, matches, customer_text, raise_window=True):
        self.current_matches = matches
        self.current_match_index = 0
        self.current_customer_text = customer_text
        self.display_current_match()
        if raise_window:
            self.response_window.deiconify()
            self.response_window.lift()
            self.response_window.attributes("-topmost", bool(SETTINGS.get("popup_on_top", True)))

    @staticmethod
    def _set_text_widget(widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def display_current_match(self):
        T = THEME
        if not self.current_matches:
            self.match_chip.configure(text="NO MATCH", bg=T["border"], fg=T["muted"])
            self.match_counter.configure(text="")
            self.match_title.configure(text="No matching objection found")
            self._set_text_widget(self.customer_quote, self.current_customer_text)
            self._set_text_widget(self.response_text, "Try a manual search or wait for more customer speech.")
            self.followup_label.configure(text="")
            return
        item = self.current_matches[self.current_match_index]
        color = category_color(item.get("category", ""))
        self.match_chip.configure(text=str(item.get("category", "")).upper() or "MATCH",
                                  bg=color, fg=text_on(color))
        count = len(self.current_matches)
        self.match_counter.configure(
            text=f"Suggestion {self.current_match_index + 1} of {count}" if count > 1 else "")
        self.match_title.configure(text=item["title"])
        self._set_text_widget(self.customer_quote, self.current_customer_text)
        self._set_text_widget(self.response_text, item["response"])
        self.followup_label.configure(text=item.get("followup") or "(No follow-up saved for this rebuttal.)")

    def next_suggestion(self):
        if self.current_matches:
            self.current_match_index = (self.current_match_index + 1) % len(self.current_matches)
            self.display_current_match()

    def mark_good_match(self):
        if not self.current_matches:
            return
        title = self.current_matches[self.current_match_index]["title"]
        self.status_var.set(f"Marked as a good match: {title}")
        self.history_text.insert("end", f"✓ Good match: {title}\n\n", "fb")
        self.history_text.see("end")

    def copy_response(self):
        if not self.current_matches:
            return
        response = self.current_matches[self.current_match_index]["response"]
        self.root.clipboard_clear()
        self.root.clipboard_append(response)
        self.status_var.set("Suggested response copied to clipboard.")
        self.copy_button.configure(text="Copied ✓")
        self.root.after(1400, lambda: self.copy_button.configure(text="Copy Response"))

    def hide_response_window(self):
        self.response_window.withdraw()

    def manual_search(self):
        query = self.search_var.get().strip()
        if not query:
            return
        matches = match_objections(query)
        if not matches:
            q = normalize(query)
            matches = [
                item for item in OBJECTIONS
                if q in normalize(item["title"]) or any(q in normalize(t) for t in item["triggers"])
            ][:4]
        if matches:
            self.show_response(matches, f"Manual search: {query}")
        else:
            messagebox.showinfo(
                "No match found",
                "No match found. Try a phrase like 'too expensive' or 'already have life insurance'.",
                parent=self.root,
            )

    # -- Transcript highlighter and saving ----------------------------------------
    def _active_transcript(self):
        return self.history_text

    def _selection_range(self):
        try:
            widget = self._active_transcript()
            return widget.index("sel.first"), widget.index("sel.last")
        except tk.TclError:
            return None

    def highlight_selection(self):
        span = self._selection_range()
        if span is None:
            self.status_var.set("Select some transcript text first, then click Highlight Selection.")
            return
        widget = self._active_transcript()
        note = self.note_var.get().strip() or NOTE_CHOICES[0]
        color_name = self.color_var.get()
        color = dict(HIGHLIGHT_COLORS).get(color_name, HIGHLIGHT_COLORS[0][1])
        self.highlight_count += 1
        tag = f"hl{self.highlight_count}"
        widget.tag_configure(tag, background=color, foreground="#101418")
        widget.tag_add(tag, *span)
        widget.tag_remove("sel", "1.0", "end")
        self.notations.append({"tag": tag, "note": note, "color": color_name})
        self.status_var.set(f'Highlighted as "{note}" ({color_name.lower()}).')

    def remove_highlight(self):
        span = self._selection_range()
        if span is None:
            self.status_var.set("Select highlighted transcript text first, then click Remove Highlight.")
            return
        widget = self._active_transcript()
        removed = 0
        for note in self.notations:
            if widget.tag_nextrange(note["tag"], *span):
                widget.tag_remove(note["tag"], *span)
                removed += 1
        self.notations = [note for note in self.notations if widget.tag_ranges(note["tag"])]
        widget.tag_remove("sel", "1.0", "end")
        self.status_var.set(f"Removed {removed} highlight(s) from the selected text."
                            if removed else "There is no highlight in the selected text.")

    def _transcript_folder(self):
        """Saved transcripts sits next to RebuttalButler.exe. If that folder is read-only
        (for example under Program Files), it goes in the data folder instead."""
        for base in dict.fromkeys([APP_DIR, DATA_DIR]):
            folder = os.path.join(base, TRANSCRIPT_FOLDER_NAME)
            if _is_writable(folder):
                return folder
        raise OSError("Could not create the Saved transcripts folder.")

    def _audio_folder(self):
        """audiofiles sits next to RebuttalButler.exe, or in the data folder if that is read-only."""
        for base in dict.fromkeys([APP_DIR, DATA_DIR]):
            folder = os.path.join(base, AUDIO_FOLDER_NAME)
            if _is_writable(folder):
                return folder
        raise OSError("Could not create the audiofiles folder.")

    @staticmethod
    def _char_offset(widget, index):
        found = widget.count("1.0", index, "chars")     # None when the index is at the very start
        return found[0] if found else 0

    def _marked_text(self, widget):
        """The transcript text with each highlighted part wrapped as <<Note: text>>."""
        body = widget.get("1.0", "end-1c")
        markers = []
        for note in self.notations:
            ranges = widget.tag_ranges(note["tag"])
            for i in range(0, len(ranges), 2):
                start = self._char_offset(widget, ranges[i])
                end = self._char_offset(widget, ranges[i + 1])
                markers.append((start, 1, f"<<{note['note']}: "))
                markers.append((end, 0, ">>"))
        pieces, previous = [], 0
        for offset, _order, marker in sorted(markers, key=lambda m: (m[0], m[1])):
            pieces.append(body[previous:offset])
            pieces.append(marker)
            previous = offset
        pieces.append(body[previous:])
        return "".join(pieces).rstrip()

    def _transcript_export(self):
        """The transcript, with each highlighted part wrapped as <<Note: text>>, then every note."""
        widget = self.history_text
        notes = []
        for note in self.notations:
            ranges = widget.tag_ranges(note["tag"])
            if not ranges:
                continue
            text = " … ".join(widget.get(ranges[i], ranges[i + 1]) for i in range(0, len(ranges), 2)).strip()
            if text:
                notes.append(f'[{note["note"]} | {note["color"]}]  "{text}"')
        lines = [
            "Rebuttal Butler transcript",
            f"Saved: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Notes: {len(notes)}",
            "",
            "TRANSCRIPT",
            "----------",
            self._marked_text(widget) or "(no speech was recorded)",
            "",
            "NOTES",
            "-----",
        ]
        lines += notes or ["(no notes)"]
        return "\n".join(lines) + "\n"

    def save_transcript(self):
        try:
            folder = self._transcript_folder()
            path = os.path.join(folder, f"Transcript {time.strftime('%Y-%m-%d %H-%M-%S')}.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(self._transcript_export())
        except OSError as exc:
            messagebox.showerror("Could not save", f"The transcript was not saved.\n\n{exc}", parent=self.root)
            return
        self.status_var.set("Transcript saved to the Saved transcripts folder.")
        messagebox.showinfo("Transcript saved", f"Saved to:\n{path}", parent=self.root)

    # -- First-run tutorial -------------------------------------------------------
    def maybe_start_tutorial(self):
        if not os.path.exists(TUTORIAL_FILE):
            self.start_tutorial()

    def start_tutorial(self):
        if self.tutorial_win is not None:
            self.tutorial_win.destroy()
            self.tutorial_win = None
        self._tutorial_step(0)

    def _tutorial_step(self, index):
        self._clear_glow()
        if index >= len(TUTORIAL_STEPS):
            self._finish_tutorial()
            return
        T = THEME
        tab, target_key, title, body = TUTORIAL_STEPS[index]
        if tab:
            self.tabbar.select(tab)
        if target_key == "popup":
            self.response_window.deiconify()
            self.response_window.lift()
        self.root.update_idletasks()
        target = self._tour_target(target_key)
        owner = target.winfo_toplevel() if target is not None else self.root

        win = self.tutorial_win = tk.Toplevel(self.root)
        win.title("Tutorial")
        set_window_icon(win)
        win.configure(bg=T["bg"])
        win.resizable(False, False)
        win.transient(owner)                        # stays above its own window, never above other programs
        win.protocol("WM_DELETE_WINDOW", lambda: self._end_tutorial(win))

        card = tk.Frame(win, bg=T["surface"], highlightthickness=1, highlightbackground=T["border"])
        card.pack(fill="both", expand=True, padx=px(18), pady=px(18))
        tk.Label(card, text=f"STEP {index + 1} OF {len(TUTORIAL_STEPS)}", font=font(8, "bold"),
                 bg=T["surface"], fg=T["accent"], anchor="w").pack(fill="x", padx=px(20), pady=(px(16), 0))
        tk.Label(card, text=title, font=font(15, "bold"), bg=T["surface"], fg=T["text"],
                 anchor="w", justify="left").pack(fill="x", padx=px(20), pady=(px(4), 0))
        tk.Label(card, text=body, font=font(11), bg=T["surface"], fg=T["text"], anchor="w", justify="left",
                 wraplength=px(430)).pack(fill="x", padx=px(20), pady=(px(10), px(16)))
        row = tk.Frame(card, bg=T["surface"])
        row.pack(fill="x", padx=px(20), pady=(0, px(16)))
        FlatButton(row, "Skip tutorial", lambda: self._end_tutorial(win), padx=12).pack(side="left")
        FlatButton(row, "OK", lambda: self._advance_tutorial(win, index), kind="primary",
                   padx=26).pack(side="right")

        win.update_idletasks()
        self._show_glow(target)
        self._place_tour_card(win, target)
        self._style_titlebar(win)
        win.lift()

    def _tour_target(self, key):
        if not key:
            return None
        if key.startswith("tab:"):
            entry = self.tabbar.items.get(key[4:])
            return entry[0] if entry else None
        return self.tour_targets.get(key)

    # Glow geometry, in pixels at 100 % scale. The rings sit outside the target and never cover it.
    _GLOW_LAYERS = ((6, 1.0), (11, 0.45), (17, 0.2))     # (distance from the target, strength)
    _GLOW_THICK = 4
    _GLOW_GOLD = "#FFD54A"

    def _show_glow(self, target):
        """A golden glow around the target. The rings are ordinary frames inside the target's own
        window, placed relative to the target with Tk's place(in_=...). Tk keeps them attached when
        the target moves, so nothing is redrawn and nothing is raised above other programs."""
        self._clear_glow()
        if target is None:
            return
        try:
            if not target.winfo_viewable():
                return
        except tk.TclError:
            return
        parent = target.winfo_toplevel()
        thick = px(self._GLOW_THICK)
        for dist_base, strength in self._GLOW_LAYERS:
            dist = px(dist_base)
            color = blend(THEME["bg"], self._GLOW_GOLD, strength)
            frames = []
            for _ in range(4):
                frames.append(tk.Frame(parent, bg=color, bd=0, highlightthickness=0))
            # placed with coordinates relative to the target's top-left corner
            w, h = target.winfo_width(), target.winfo_height()
            placements = [
                dict(x=-dist, y=-dist, width=w + 2 * dist, height=thick),                 # top
                dict(x=-dist, y=h + dist - thick, width=w + 2 * dist, height=thick),     # bottom
                dict(x=-dist, y=-dist, width=thick, height=h + 2 * dist),                # left
                dict(x=w + dist - thick, y=-dist, width=thick, height=h + 2 * dist),     # right
            ]
            for frame, spot in zip(frames, placements):
                frame.place(in_=target, **spot)
                frame.lift()
                self._tour_glow.append(frame)
        self._tour_target_widget = target

    def _tour_spots(self, target, cw, ch):
        """Where the tour window goes for each corner. The window has an 18 px border around the
        card, so the card's own edge, not the border, is what lines up with the glow."""
        x0, y0, x1, y1 = self._glow_outline(target)
        gap, pad = px(12), px(18)
        return {
            "right-top": (x1 + gap - pad, y0 - pad),
            "right-bottom": (x1 + gap - pad, y1 - ch + pad),
            "left-top": (x0 - gap - cw + pad, y0 - pad),
            "left-bottom": (x0 - gap - cw + pad, y1 - ch + pad),
            "below": (x0 - pad, y1 + gap - pad),
            "above": (x0 - pad, y0 - gap - ch + pad),
        }

    def _glow_outline(self, target):
        """The outside edge of the glow, in screen coordinates."""
        outer = px(self._GLOW_LAYERS[-1][0])          # the outermost ring's outer edge
        x, y = target.winfo_rootx(), target.winfo_rooty()
        return x - outer, y - outer, x + target.winfo_width() + outer, y + target.winfo_height() + outer

    def _place_tour_card(self, win, target):
        """Pop the card out from a corner of the glow. Right-top is tried first, then right-bottom,
        left-top and left-bottom. The card's own corner sits on the glow's corner, so it lines up
        with the highlight. Falls back to above or below, then the centre of the screen."""
        self._tour_corner = None
        cw, ch = win.winfo_reqwidth(), win.winfo_reqheight()
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        if target is None or not target.winfo_viewable():
            x = self.root.winfo_rootx() + max(0, (self.root.winfo_width() - cw) // 2)
            y = self.root.winfo_rooty() + px(110)
            win.geometry(f"+{int(x)}+{int(y)}")
            return
        spots = self._tour_spots(target, cw, ch)
        for name in ("right-top", "right-bottom", "left-top", "left-bottom", "below", "above"):
            x, y = spots[name]
            if x >= 0 and y >= 0 and x + cw <= sw and y + ch <= sh:
                self._tour_corner = name
                win.geometry(f"+{int(x)}+{int(y)}")
                return
        x, y = max(0, (sw - cw) // 2), max(0, (sh - ch) // 2)
        win.geometry(f"+{int(x)}+{int(y)}")

    def _on_tour_configure(self, _event=None):
        """A window that holds the target moved or resized: keep the card on its corner."""
        if self.tutorial_win is None or self._tour_corner is None or self._tour_target_widget is None:
            return
        if _event is not None and str(getattr(_event, "widget", "")) not in (str(self.root), str(self.response_window)):
            return
        try:
            target = self._tour_target_widget
            if not target.winfo_viewable():
                return
            cw, ch = self.tutorial_win.winfo_reqwidth(), self.tutorial_win.winfo_reqheight()
            x, y = self._tour_spots(target, cw, ch)[self._tour_corner]
            self.tutorial_win.geometry(f"+{int(x)}+{int(y)}")
        except tk.TclError:
            pass

    def _clear_glow(self):
        for frame in self._tour_glow:
            try:
                frame.destroy()
            except tk.TclError:
                pass
        self._tour_glow = []
        self._tour_corner = None
        self._tour_target_widget = None

    def _advance_tutorial(self, win, index):
        win.destroy()
        self.tutorial_win = None
        self._tutorial_step(index + 1)

    def _end_tutorial(self, win):
        self._clear_glow()
        win.destroy()
        self.tutorial_win = None
        self._finish_tutorial()

    @staticmethod
    def _finish_tutorial():
        try:
            with open(TUTORIAL_FILE, "w", encoding="utf-8") as handle:
                handle.write("done\n")
        except OSError:
            pass

    def close_app(self):
        self.engine.stop()
        self.root.destroy()


LOG_FILE = os.path.join(DATA_DIR, "mrai.log")


def _setup_logging():
    """The packaged app has no console, so anything it prints would vanish. Keep it in
    mrai.log next to the library so problems can be looked at afterwards."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    try:
        if os.path.isfile(LOG_FILE) and os.path.getsize(LOG_FILE) > 1_000_000:
            os.remove(LOG_FILE)
        log = open(LOG_FILE, "a", buffering=1, encoding="utf-8")
        log.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    except OSError:
        log = open(os.devnull, "w")
    if sys.stdout is None:
        sys.stdout = log
    if sys.stderr is None:
        sys.stderr = log


def run_selftest():
    """Check the things that go wrong on a fresh PC. -> (all_ok, report_text).
    Run it from a shell as `RebuttalButler.exe --selftest` or `python main_standalone.py --selftest`."""
    lines = []
    ok = True

    def say(mark, text):
        nonlocal ok
        if mark == "FAIL":
            ok = False
        lines.append(f"[{mark}] {text}")

    lines.append(f"Rebuttal Butler self-test ({'packaged app' if FROZEN else 'script'}, Python {sys.version.split()[0]})")
    lines.append(f"App folder:  {APP_DIR}")
    lines.append(f"Data folder: {DATA_DIR}")
    lines.append("")
    say("ok", f"Rebuttal library: {len(OBJECTIONS)} objections loaded")
    if os.path.isfile(ICON_FILE) and os.path.isfile(LOGO_FILE):
        say("ok", "Branding: logo and icon files are present")
    else:
        say("warn", "Branding: logo or icon missing from the assets folder (the app still works without them)")
    # 1. speech engine and its voice filter
    try:
        import ctranslate2
        import onnxruntime
        from faster_whisper.utils import get_assets_path
        assets = get_assets_path()
        vad = [f for f in os.listdir(assets) if f.endswith(".onnx")]
        if not vad:
            raise RuntimeError(f"voice-filter file missing from {assets}")
        from faster_whisper.vad import get_speech_timestamps
        get_speech_timestamps(np.zeros(SAMPLE_RATE, dtype=np.float32))       # runs the voice filter once
        say("ok", f"Speech engine: CTranslate2 {ctranslate2.__version__}, voice filter {vad[0]} runs")
    except Exception as exc:
        say("FAIL", f"Speech engine problem: {exc}")
    # 2. graphics: is an NVIDIA GPU usable?
    if _DEVICE_PREF == "cpu":
        say("ok", "Graphics: MRAI_DEVICE=cpu, Whisper runs on the CPU")
    else:
        usable, why = _gpu_status()
        if usable:
            say("ok", f"Graphics: {why}. Whisper will run on the GPU (CPU is the fallback).")
        else:
            say("warn", f"Graphics: {why}. Whisper runs on the CPU, which works but is slower.")
    # 3. the speech model
    folder = local_model_dir(MODEL_SIZE)
    if folder:
        try:
            started = time.monotonic()
            LiveTranscriber(queue.Queue())._make_model(MODEL_SIZE, "cpu", COMPUTE_TYPE if DEVICE == "cpu" else "int8")
            say("ok", f"Whisper model '{MODEL_SIZE}' is bundled ({folder}) and runs; loading took {time.monotonic() - started:.1f}s")
        except Exception as exc:
            say("FAIL", f"Whisper model '{MODEL_SIZE}' is in {folder} but could not be loaded: {exc}")
    else:
        say("warn", f"Whisper model '{MODEL_SIZE}' is not in a models folder ({', '.join(MODEL_DIRS)}). "
                    "It will be downloaded on first run, which needs the internet.")
    # 3. desktop audio
    default_name, outputs = desktop_audio_info()
    if pyaudio is None:
        say("warn", "Desktop audio is unavailable (PyAudioWPatch missing, or this is not Windows). Inputs only.")
    elif not outputs:
        say("FAIL", "Desktop audio: no speaker or headset output found. Check Windows sound settings.")
    else:
        say("ok", f"Desktop audio: default output '{default_name or 'unknown'}', {len(outputs)} output(s) found")
        try:
            blocks, peak = LiveTranscriber(queue.Queue()).probe(("desktop", None), 2.0)
            if blocks == 0:
                say("warn", "Desktop audio opened but received nothing in 2 s. Normal if the PC was silent: "
                            "play any sound while running the self-test to see a level.")
            else:
                say("ok", f"Desktop audio capture works ({blocks} blocks, peak level {peak:.3f})")
        except Exception as exc:
            say("FAIL", f"Desktop audio could not be opened: {exc}")
    # 4. microphones / cables
    if sd is None:
        say("warn", "Microphone support (sounddevice) is not available.")
    else:
        try:
            count = sum(1 for d in sd.query_devices() if int(d.get("max_input_channels", 0)) > 0)
            say("ok", f"Microphones and cables: {count} input device(s)")
        except Exception as exc:
            say("warn", f"Could not list microphones: {exc}")
    lines.append("")
    lines.append("RESULT: " + ("everything needed is present" if ok else "something above needs fixing"))
    return ok, "\n".join(lines)


def _report_crash():
    details = traceback.format_exc()
    try:
        print("\n" + "=" * 65 + "\nREBUTTAL BUTLER CRASHED\n" + "=" * 65)
        print(details)
    except Exception:
        pass
    try:
        with open(os.path.join(DATA_DIR, "crash.txt"), "w", encoding="utf-8") as handle:
            handle.write(details)
    except OSError:
        pass
    try:
        window = tk.Tk()
        window.withdraw()
        messagebox.showerror("Rebuttal Butler crashed",
                             f"Something went wrong:\n\n{details[-1500:]}\n\nThis was also saved to crash.txt in {DATA_DIR}.")
        window.destroy()
    except Exception:
        pass


def main():
    _setup_logging()
    if "--selftest" in sys.argv:
        ok, report = run_selftest()
        print(report)
        try:
            with open(os.path.join(DATA_DIR, "selftest.txt"), "w", encoding="utf-8") as handle:
                handle.write(report + "\n")
        except OSError:
            pass
        if FROZEN and not sys.stdin:
            window = tk.Tk()
            window.withdraw()
            (messagebox.showinfo if ok else messagebox.showwarning)("Self-test", report)
            window.destroy()
        sys.exit(0 if ok else 1)
    _enable_dpi_awareness()
    root = tk.Tk()
    app = MortgageRebuttalApp(root)
    root.after(900, app.maybe_start_tutorial)     # first run only: the tour opens once the window is up
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        _report_crash()
        if not FROZEN and sys.stdin is not None and sys.stdin.isatty():
            input("\nPress Enter to close this window...")
