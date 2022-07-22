# Multimodal query and prediction

## Dependencies

- python3

## Data

- [Raw data](http://insideairbnb.com/get-the-data/)
- [Processed data](https://drive.google.com/drive/folders/1eZnSlu-7zhKRUrZiOBEcJOFhY4GlOuGH?usp=sharing)

## Command

```
python3 main.py --epoch 1000 --amenLen 100 \
        --path model.pt --batch_size 1024  \
        --loaddata --mode [train, infer]
```
