# Pre-trained weights

| File | Size | Description |
|---|---|---|
| `yolo11l_soccer.pt` | 51 MB | YOLO11-L person detector (players, goalkeepers, referees), fine-tuned on SoccerNet tracking at 1920 px. Epoch-7 checkpoint of the run in `results/detector_training/`. Optimizer state stripped to fit GitHub's file-size limit (the original `best.pt` was 107 MB); it produced exactly the same boxes and scores as the original on the frame I compared. |
| `osnet_ain_x1_0_soccer.pt` | 11 MB | OSNet-AIN x1.0 ReID model fine-tuned on SoccerNet identity crops (epoch-40 checkpoint), exported as a plain `state_dict`. BoxMOT infers the architecture from the file name, so keep `osnet_ain_x1_0` in it. |

SHA-256:

```
d26347960387e755e4aa1bb11df0b4d80adfcc3743ef074eab8ae6212600df19  yolo11l_soccer.pt
2f2783a4e242804a35c89b9024a9acc8011134c3b5957c32325460e8dffdbe28  osnet_ain_x1_0_soccer.pt
```

The weights were trained on [SoccerNet](https://www.soccer-net.org/) data and inherit its research-use terms.
