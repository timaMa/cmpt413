# Output results and evalutaion code

All the output files are in the directory "output". <br />
All the references are in the directory "references" <br />
"get_score.py" contains the evaluation code.<br />
"baseline_output.txt" is the output file after directly running the baseline method. (not related to the output results. Only used in notebook.)

## How to run the code 

```
python3 get_score.py <method name> <dataset name>
```

Examples (including all possible arguments):

```
python3 get_score.py "baseline" "validation"
python3 get_score.py "baseline" "test"
python3 get_score.py "siamese" "train"
python3 get_score.py "siamese" "validation"
python3 get_score.py "siamese" "test"
```

There is no "baseline" "train".
