#!/usr/local/lib/python3.5/dist-packages
import formatData
from ../../models/ import alexNet_2

Xtr, Ytr, Xte, Yte = formatData.knownEnv(formatData.load_data(),formatData.load_poses(), training_ratio=(7/10.0))
alexNet.run_model(alexNet.create_model(), Xtr, Ytr, Xte, Yte)
