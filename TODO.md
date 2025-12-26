1) Fix steering model bake-in / export. It returned a 500 error: 127.0.0.1 - - [26/Dec/2025 05:17:03] "POST /api/apply-steering-permanent HTTP/1.1" 500 -
2) Add color visuals to features ranking to indicate harmful and harmless strength for each feature
3) When a vectors are backed into a model, and the model is exported, include an export of the configurations used to steer the model
