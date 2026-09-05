#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


# ============================================================
# PHASE 2
# AI BASED NETWORK INTRUSION DETECTION SYSTEM
# ============================================================

import time
import json
import joblib
import warnings

import numpy as np
import pandas as pd

from collections import defaultdict
from scapy.all import sniff, IP, TCP, UDP

warnings.filterwarnings("ignore")

print("=" * 60)
print("PHASE 2 - REAL TIME NETWORK INTRUSION DETECTION")
print("=" * 60)

print("Libraries Imported Successfully")


# # Load Trained Model Files

# In[7]:


# ============================================================
# LOAD TRAINED MODEL FILES
# ============================================================

import os
import json
import joblib

print("=" * 60)
print("LOADING TRAINED MODEL FILES")
print("=" * 60)

PHASE1_PATH = os.path.join("..", "Phase1")



model = joblib.load(
    os.path.join(PHASE1_PATH, "ids_model.pkl")
)

scaler = joblib.load(
    os.path.join(PHASE1_PATH, "scaler.pkl")
)

label_encoder = joblib.load(
    os.path.join(PHASE1_PATH, "label_encoder.pkl")
)

with open(
    os.path.join(PHASE1_PATH, "feature_columns.json"),
    "r"
) as f:
    feature_columns = json.load(f)

print("✓ Random Forest Model Loaded")
print("✓ Label Encoder Loaded")
print("✓ Feature Columns Loaded")
print()
print("Total Features :", len(feature_columns))


# # Live Packet Capture

# In[8]:


# ============================================================
# CELL 3
# LIVE PACKET CAPTURE USING SCAPY
# ============================================================

from scapy.all import sniff

print("=" * 60)
print("STARTING PACKET CAPTURE")
print("=" * 60)


# Packet storage
packet_buffer = []


# Packet callback function
def packet_callback(packet):

    packet_buffer.append(packet)

    print(
        f"Packet Captured : {len(packet_buffer)}"
    )


# Capture packets
sniff(
    prn=packet_callback,
    count=20,
    store=False
)


print()
print("=" * 60)
print("PACKET CAPTURE COMPLETED")
print("=" * 60)

print(
    "Total Packets Captured :",
    len(packet_buffer)
)


# In[9]:


# Display first captured packet

packet_buffer[0]


# # Flow Manager

# In[10]:


# ============================================================
# CELL 4
# FLOW MANAGER
# ============================================================

from collections import defaultdict
import time


print("=" * 60)
print("FLOW MANAGER INITIALIZATION")
print("=" * 60)


class Flow:

    def __init__(self, packet):

        self.start_time = time.time()
        self.end_time = time.time()

        self.src_ip = None
        self.dst_ip = None

        self.src_port = 0
        self.dst_port = 0

        self.protocol = 0

        self.forward_packets = 0
        self.backward_packets = 0

        self.forward_bytes = 0
        self.backward_bytes = 0

        self.packet_lengths = []

        self.timestamps = []

        self.flags = []


        self.update(packet)


    def update(self, packet):

        self.end_time = time.time()

        self.timestamps.append(
            self.end_time
        )


        # IP Layer

        if packet.haslayer("IP"):

            self.src_ip = packet["IP"].src
            self.dst_ip = packet["IP"].dst


            self.protocol = packet["IP"].proto



        # TCP

        if packet.haslayer("TCP"):

            self.src_port = packet["TCP"].sport
            self.dst_port = packet["TCP"].dport


            self.flags.append(
                int(packet["TCP"].flags)
            )


        # UDP

        elif packet.haslayer("UDP"):

            self.src_port = packet["UDP"].sport
            self.dst_port = packet["UDP"].dport



        length = len(packet)


        self.packet_lengths.append(
            length
        )


        self.forward_packets += 1

        self.forward_bytes += length



# ============================================================
# FLOW STORAGE
# ============================================================


flow_table = {}


def get_flow_key(packet):

    if packet.haslayer("IP"):

        src = packet["IP"].src
        dst = packet["IP"].dst

    else:
        return None


    sport = 0
    dport = 0


    if packet.haslayer("TCP"):

        sport = packet["TCP"].sport
        dport = packet["TCP"].dport


    elif packet.haslayer("UDP"):

        sport = packet["UDP"].sport
        dport = packet["UDP"].dport



    protocol = packet["IP"].proto


    return (
        src,
        dst,
        sport,
        dport,
        protocol
    )



# ============================================================
# PROCESS CAPTURED PACKETS
# ============================================================


for pkt in packet_buffer:

    key = get_flow_key(pkt)


    if key is None:
        continue


    if key not in flow_table:

        flow_table[key] = Flow(pkt)


    else:

        flow_table[key].update(pkt)



print()
print("=" * 60)
print("FLOW CREATION COMPLETED")
print("=" * 60)


print(
    "Total Flows Created :",
    len(flow_table)
)


# In[11]:


list(flow_table.keys())[:5]


# In[12]:


first_flow = list(flow_table.values())[0]

first_flow.__dict__


# # Feature Extraction

# In[13]:


# ============================================================
# CELL 5
# FEATURE EXTRACTION - 55 FEATURES
# ============================================================

import numpy as np
import pandas as pd


print("=" * 60)
print("FEATURE EXTRACTION")
print("=" * 60)



def extract_features(flow):


    features = {}


    # --------------------------------------------------------
    # Basic Flow Information
    # --------------------------------------------------------

    duration = (
        flow.end_time - flow.start_time
    )


    packet_lengths = flow.packet_lengths


    if len(packet_lengths) == 0:

        packet_lengths = [0]


    packet_lengths = np.array(
        packet_lengths
    )


    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    features["Destination Port"] = flow.dst_port


    features["Flow Duration"] = duration


    features["Total Fwd Packets"] = (
        flow.forward_packets
    )


    features["Total Backward Packets"] = (
        flow.backward_packets
    )


    features["Total Length of Fwd Packets"] = (
        flow.forward_bytes
    )


    features["Total Length of Bwd Packets"] = (
        flow.backward_bytes
    )



    # Packet statistics

    features["Fwd Packet Length Max"] = (
        packet_lengths.max()
    )


    features["Fwd Packet Length Min"] = (
        packet_lengths.min()
    )


    features["Fwd Packet Length Mean"] = (
        packet_lengths.mean()
    )


    features["Fwd Packet Length Std"] = (
        packet_lengths.std()
        if len(packet_lengths) > 1
        else 0
    )


    features["Bwd Packet Length Max"] = 0

    features["Bwd Packet Length Min"] = 0

    features["Bwd Packet Length Mean"] = 0

    features["Bwd Packet Length Std"] = 0



    # Packet length

    features["Min Packet Length"] = (
        packet_lengths.min()
    )


    features["Max Packet Length"] = (
        packet_lengths.max()
    )


    features["Packet Length Mean"] = (
        packet_lengths.mean()
    )


    features["Packet Length Std"] = (
        packet_lengths.std()
        if len(packet_lengths)>1
        else 0
    )


    features["Packet Length Variance"] = (
        packet_lengths.var()
        if len(packet_lengths)>1
        else 0
    )


    features["Average Packet Size"] = (
        packet_lengths.mean()
    )



    # --------------------------------------------------------
    # Rate Features
    # --------------------------------------------------------

    if duration > 0:

        features["Flow Bytes/s"] = (
            (flow.forward_bytes + flow.backward_bytes)
            /
            duration
        )

        features["Flow Packets/s"] = (
            (flow.forward_packets + flow.backward_packets)
            /
            duration
        )


        features["Fwd Packets/s"] = (
            flow.forward_packets / duration
        )


        features["Bwd Packets/s"] = (
            flow.backward_packets / duration
        )

    else:

        features["Flow Bytes/s"] = 0

        features["Flow Packets/s"] = 0

        features["Fwd Packets/s"] = 0

        features["Bwd Packets/s"] = 0



    # --------------------------------------------------------
    # Remaining CIC Features
    # --------------------------------------------------------

    remaining_features = [

        "Flow IAT Mean",
        "Flow IAT Std",
        "Flow IAT Max",
        "Flow IAT Min",

        "Fwd IAT Mean",
        "Fwd IAT Std",
        "Fwd IAT Max",
        "Fwd IAT Min",

        "Bwd IAT Mean",
        "Bwd IAT Std",
        "Bwd IAT Max",
        "Bwd IAT Min",

        "Fwd IAT Total",
        "Bwd IAT Total",

        "FIN Flag Count",
        "SYN Flag Count",
        "RST Flag Count",
        "PSH Flag Count",
        "ACK Flag Count",
        "URG Flag Count",
        "ECE Flag Count",
        "CWE Flag Count",

        "Fwd PSH Flags",
        "Bwd PSH Flags",

        "Fwd URG Flags",
        "Bwd URG Flags",

        "Fwd Header Length",
        "Bwd Header Length",

        "Avg Fwd Segment Size",
        "Avg Bwd Segment Size",

        "Down/Up Ratio"

    ]


    for f in remaining_features:

        features[f] = 0



    return features



# ============================================================
# APPLY FEATURE EXTRACTION TO ALL FLOWS
# ============================================================


feature_list = []


for key, flow in flow_table.items():

    feature_list.append(
        extract_features(flow)
    )



features_df = pd.DataFrame(
    feature_list
)



print()

print("Feature Extraction Completed")

print()

print(
    "Number of Flows :",
    len(features_df)
)


print(
    "Number of Features :",
    features_df.shape[1]
)


features_df.head()


# In[14]:


features_df.columns.tolist()


# # FEATURE VALIDATION

# In[16]:


# ============================================================
# FEATURE VALIDATION
# COMPARE LIVE FEATURES VS TRAINING FEATURES
# ============================================================

import os
import json


print("=" * 60)
print("FEATURE COMPARISON")
print("=" * 60)


# ============================================================
# LOAD TRAINING FEATURES FROM PHASE1
# ============================================================

PHASE1_PATH = os.path.join("..", "Phase1")


with open(
    os.path.join(PHASE1_PATH, "feature_columns.json"),
    "r"
) as f:
    training_features = json.load(f)



# ============================================================
# LIVE GENERATED FEATURES
# ============================================================

live_features = features_df.columns.tolist()



print()

print("Training Feature Count :", len(training_features))

print("Live Feature Count     :", len(live_features))


# ============================================================
# MISSING FEATURES
# ============================================================

missing_features = list(
    set(training_features) - set(live_features)
)


print()
print("=" * 60)
print("MISSING FEATURES")
print("=" * 60)


print("Missing Count :", len(missing_features))


for i, feature in enumerate(missing_features, start=1):
    print(f"{i} : {feature}")



# ============================================================
# EXTRA FEATURES
# ============================================================

extra_features = list(
    set(live_features) - set(training_features)
)


print()

print("=" * 60)
print("EXTRA FEATURES")
print("=" * 60)


print("Extra Count :", len(extra_features))


for i, feature in enumerate(extra_features, start=1):
    print(f"{i} : {feature}")



# ============================================================
# ORDER CHECK
# ============================================================

print()

print("=" * 60)
print("FEATURE ORDER CHECK")
print("=" * 60)


if training_features == live_features:

    print("✓ Feature order is correct")

else:

    print("✗ Feature order mismatch")



# ============================================================
# FINAL STATUS
# ============================================================

print()

if (
    len(missing_features) == 0
    and
    len(extra_features) == 0
    and
    training_features == live_features
):

    print("STATUS : READY FOR MODEL")

else:

    print("STATUS : FEATURE FIX REQUIRED")


# # Live Prediction

# In[19]:


# ============================================================
# CELL 7
# LIVE PREDICTION
# ============================================================


import numpy as np
import pandas as pd


print("="*60)
print("LIVE ATTACK PREDICTION")
print("="*60)



# ------------------------------------------------------------
# Arrange Features
# ------------------------------------------------------------

X_live = features_df[
    feature_columns
]


print("Input Shape :", X_live.shape)



# ------------------------------------------------------------
# Scaling
# ------------------------------------------------------------

X_scaled = scaler.transform(
    X_live
)


print("Feature Scaling Completed")



# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------


prediction = model.predict(
    X_scaled
)


probability = model.predict_proba(
    X_scaled
)



# ------------------------------------------------------------
# Convert Label
# ------------------------------------------------------------

attack_prediction = label_encoder.inverse_transform(
    prediction
)



confidence = np.max(
    probability,
    axis=1
)



# ------------------------------------------------------------
# Result DataFrame
# ------------------------------------------------------------


prediction_df = pd.DataFrame(
    {

        "Flow_ID":
        range(1,len(attack_prediction)+1),


        "Prediction":
        attack_prediction,


        "Confidence (%)":
        np.round(confidence*100,2)

    }
)



print()

print("="*60)
print("PREDICTION RESULT")
print("="*60)


prediction_df


# # Alert Generation

# In[20]:


# ============================================================
# CELL 8
# ALERT GENERATION
# ============================================================


from datetime import datetime
import pandas as pd


print("=" * 60)
print("GENERATING ALERTS")
print("=" * 60)



alerts = []



for index, row in prediction_df.iterrows():


    attack = row["Prediction"]

    confidence = row["Confidence (%)"]



    # Current Time

    timestamp = datetime.now()



    # Decide Status

    if attack == "BENIGN":

        status = "NORMAL"


    else:

        status = "ALERT"



    alerts.append(

        {

            "Time": timestamp,

            "Flow_ID": row["Flow_ID"],

            "Status": status,

            "Attack": attack,

            "Confidence (%)": confidence

        }

    )



# Convert to DataFrame

alert_df = pd.DataFrame(
    alerts
)



print()

print("=" * 60)
print("ALERT GENERATION COMPLETED")
print("=" * 60)


alert_df


# In[21]:


print()
print("="*60)
print("ALERT SUMMARY")
print("="*60)


print(
    alert_df["Status"].value_counts()
)


print()


print(
    alert_df["Attack"].value_counts()
)


# # Attack Logger

# In[22]:


# ============================================================
# CELL 9
# ATTACK LOGGER
# ============================================================


import os
import pandas as pd


print("=" * 60)
print("ATTACK LOGGER")
print("=" * 60)



# ============================================================
# CREATE LOG DIRECTORY
# ============================================================


LOG_FOLDER = "logs"


if not os.path.exists(LOG_FOLDER):

    os.makedirs(LOG_FOLDER)



LOG_FILE = os.path.join(
    LOG_FOLDER,
    "nids_attack_logs.csv"
)



# ============================================================
# SAVE ALERT DATA
# ============================================================


if os.path.exists(LOG_FILE):

    # Existing logs

    old_logs = pd.read_csv(
        LOG_FILE
    )


    updated_logs = pd.concat(
        [
            old_logs,
            alert_df
        ],
        ignore_index=True
    )


else:

    # First time logging

    updated_logs = alert_df



# Save

updated_logs.to_csv(
    LOG_FILE,
    index=False
)



print()

print("=" * 60)
print("LOGGING COMPLETED")
print("=" * 60)


print(
    "Log File :",
    LOG_FILE
)


print()

print(
    "Total Records :",
    len(updated_logs)
)



# Display latest logs

print()

print("=" * 60)
print("LATEST ATTACK LOGS")
print("=" * 60)


updated_logs.tail(10)


# # Real-Time Detection Loop

# In[24]:


# ============================================================
# CELL 10
# REAL-TIME DETECTION LOOP
# ============================================================


import time
import pandas as pd



print("="*60)
print("REAL TIME NIDS STARTED")
print("="*60)



def run_detection_cycle():


    print()
    print("-"*60)

    print("Processing New Traffic...")


    # --------------------------------------------------------
    # Step 1
    # Extract Features From Current Flows
    # --------------------------------------------------------

    live_features = []


    for key, flow in flow_table.items():


        feature = extract_features(flow)


        live_features.append(
            feature
        )



    if len(live_features) == 0:

        print("No flows detected")

        return



    features_df_live = pd.DataFrame(
        live_features
    )



    print(
        "Flows Detected :",
        len(features_df_live)
    )



    # --------------------------------------------------------
    # Step 2
    # Arrange Feature Order
    # --------------------------------------------------------


    X_live = features_df_live[
        feature_columns
    ]

    



    # --------------------------------------------------------
    # Step 3
    # Scaling
    # --------------------------------------------------------


    X_scaled = scaler.transform(
        X_live
    )



    # --------------------------------------------------------
    # Step 4
    # Prediction
    # --------------------------------------------------------


    prediction = model.predict(
        X_scaled
    )


    probability = model.predict_proba(
        X_scaled
    )


    attack_names = label_encoder.inverse_transform(
        prediction
    )


    confidence = (
        probability.max(axis=1)
        *
        100
    )



    # --------------------------------------------------------
    # Step 5
    # Create Result
    # --------------------------------------------------------


    detection_result = pd.DataFrame(

        {

            "Flow_ID":
            range(
                1,
                len(attack_names)+1
            ),


            "Prediction":
            attack_names,


            "Confidence (%)":
            confidence.round(2)

        }

    )



    print()

    print(
        detection_result
    )



    return detection_result




# ============================================================
# CONTINUOUS LOOP
# ============================================================


while True:


    result = run_detection_cycle()



    if result is not None:


        print()

        print(
            "Detection Cycle Completed"
        )



    time.sleep(10)





