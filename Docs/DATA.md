```
# Download the file
wget "https://www.dropbox.com/scl/fi/6numm9wzu1cixw8nyzb91/all_sqe.zip?rlkey=it1ruxtsm4rggxldbbbr4w3yj&e=1&dl=1" -O data/NW-UCLA/all_sqe.zip

# Extract it
cd data/NW-UCLA
unzip all_sqe.zip
cd ../..

```


# NTU RGB+D 60
## 1. Raw skeleton
- download zip file from the dataset websites

- put `nturgbd_skeletons_s001_to_s017.zip` inside /data/ntu

- unzip by: `unzip -q ./nturgbd_skeletons_s001_to_s017.zip -d ./ `

- run `python get_raw_skes_data.py`, get:
```
...
Saved raw bodies data into ./raw_data/raw_skes_data.pkl
Total frames: 4773093
```

## 2. Denoised skeleton
- data processed above, so not gonna download/move any files

- run 
```
python get_raw_denoised_data.py
```
and get 
```
...
Processing S017C003P020R002A059
Processing S017C003P020R002A060
Saved raw denoised positions of 4772570 frames into ./denoised_data/raw_denoised_joints.pkl
Found 138 files that have missing data
```