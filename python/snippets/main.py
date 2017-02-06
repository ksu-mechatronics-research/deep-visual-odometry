#!/usr/local/lib/python3.5/dist-packages
import formatData
import alexNet

Xtr, Ytr, Xte, Yte = formatData.knownEnv(formatData.load_data(),formatData.load_poses())
alexNet.run_model(alexNet.create_model(), Xtr, Ytr, Xte, Yte)
