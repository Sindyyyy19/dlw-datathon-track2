# Q&A sheet for the judges (Track 2, fraud detection)

Plain-English answers to the questions judges are likely to ask. Numbers are from the notebook.

**What is the problem, in one sentence?**
Rank 12,000 transactions so the few fraudulent ones come out on top, then pick a cut-off that catches as much fraud as possible without bothering too many real customers.

**Why not accuracy?**
Only 1.8% of transactions are fraud. A model that says "never fraud" is 98.2% accurate and useless. We use PR-AUC, which measures how well fraud is ranked above everything else, and then F1 and recall at a cut-off we chose on purpose.

**What did you find in the data?**
Six weak signals, no strong one: a new device (7.5% fraud vs 1.2%), the merchant category (luxury, cash transfer, electronics are 3 to 5 times riskier than groceries), a foreign country, bank transfer or e-commerce channels, late night, and bigger amounts on younger, busier accounts. Also: the test set has twice as many missing values as the training set, so the model had to handle gaps without crashing.

**How did you clean the data?**
Nothing was dropped. Missing values are given to the tree model as missing, filled with the median for the linear model, and every column gets a "was this missing" flag, because a missing value is itself slightly suspicious.

**Which features did you build, and why?**
Ratios that compare this payment with the account's own recent behaviour (this amount versus the average of the last 24 hours, the share of today's spend, the share of today's transactions in the last hour), plus explicit flags for night, foreign, new account and high-risk category, and two combinations: new device abroad, new device in a high-risk category.

**Which model, and why that one?**
A blend of a regularised logistic regression and a very small LightGBM (3 leaves, 200 trees). With only 353 fraud examples, flexible models memorise noise: our big LightGBM scored worse than the plain logistic baseline. The two small models make different mistakes, so averaging them is a little better than either alone.

**How did you validate?**
Five-fold stratified cross-validation (each fold keeps the 1.8% fraud rate). Every number we report is on data the model had not seen.

**What are the scores?**
PR-AUC 0.21 (random would be 0.018, so about 12 times better), ROC-AUC 0.78. At our cut-off: recall 34%, precision 20%, F1 0.25, with 3% of transactions flagged. If you prefer the best F1 instead, it is 0.29 with recall 25%.

**Why that cut-off?**
We maximised F1.5, which weights recall a bit more than precision, because a missed fraud usually costs more than one confirmation message to a customer. The threshold is a business dial: the notebook shows the full table so it can be moved.

**What are the model's assumptions?**
The hidden test data comes from the same process; each transaction is independent (there is no customer id); the 24-hour and 1-hour counts were computed before the transaction, so they are not leakage.

**What are its limits?**
The signal in these columns is weak: about one in five of our top flags is real fraud, and every model family lands in the same band. With 353 positives the exact threshold is uncertain by a few points of recall.

**What would you do next?**
Customer or card identifiers (that is where real fraud systems get their lift), a cost-based threshold using real loss and friction numbers, calibrated probabilities for a tiered response (allow, confirm by SMS, block), and drift monitoring on missingness and country mix.

**Honest answer if asked "is 0.21 good?"**
It is modest in absolute terms and we say so. The point of the notebook is that the choice of model, the validation and the threshold are right for a weak-signal, heavily imbalanced problem; a higher public leaderboard score from a bigger model would most likely not survive the private set.
