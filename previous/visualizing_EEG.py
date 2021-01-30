"""import mne

raw = mne.io.read_raw_edf("C:/Users/user/Desktop/AI/EEG_classification-main/EEG/data/SC4001E0-PSG.edf", preload=True)
mne.set_log_level("WARNING")
print("raw: ")
print(raw)
print("raw info: ")
print(raw.info)
print("rwa info [sfreq]")
print(raw.info["sfreq"])

raw.rename_channels(lambda s: s.strip("."))
print(mne.channels.get_builtin_montages())
mne.channels.get_builtin_montages()
montage = mne.channels.make_standard_montage("EGI_256") # "standard_1020"
montage.plot()
#raw.plot()"""

import mne


mne.set_log_level("WARNING")
raw = mne.io.read_raw_edf("C:/Users/user/Desktop/AI/EEG_classification-main/EEG/data/SC4001E0-PSG.edf", preload=True)

raw.rename_channels(lambda s: s.strip("."))
# print(mne.channels.get_builtin_montages())


mne.channels.make_standard_montage("EGI_256")
#raw.set_eeg_reference("average")

raw.plot()