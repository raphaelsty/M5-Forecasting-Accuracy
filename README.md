**Hi,**

Deploying and maintaining machine learning algorithms in a **production** environment is not an easy task. The **drift** of data over the time tends to degrade the performance of the algorithms because the models are static. Data Scientist **re-train models from scratch** to update them. This task is tedious and monopolizes highly qualified human resources. 

**I would like to present a solution to these problems**. I will use online learning and the open-source **[Creme](https://github.com/creme)** library (I am a core developer of Creme) to overcome the difficulties of deploying a machine learning model in production. 

I will illustrate my point with data from the **[M5-Forecasting-Accuracy kaggle competition](https://www.kaggle.com/c/m5-forecasting-accuracy/)** which is well suited to the use case of Creme. **The objective of the M5-Forecasting-Accuracy competition is to estimate the daily sales of 30490 products for the next 28 days.**

My goal is not to develop a competitive model, but to show the simplicity of an online learning model for an event-based dataset such as M5-Forecasting-Accuracy.

First of all, I would like to share with you the deployment process I follow when I need to deploy a machine learning algorithm such as the LightGBM model or scikit-learn models in a production environment for a task similar to the M5-Forecasting-Accuracy competition. 

I will then compare this process to **deploying a model** with the Creme and Chantilly libraries.

[Max Halford](https://maxhalford.github.io) is the main developer of Creme and he's the one who initiated the project, he did a blog post **[here](https://towardsdatascience.com/machine-learning-for-streaming-data-with-creme-dacf5fb469df)**. This is a good introduction to the philosophy of online learning and especially Creme philosophy. Feel free to have a look at it if you are interested in the subject. 

![](static/creme.svg)

### Model deployment when fitting data in one row:

Deploying a model that learns by batch requires a well-oiled organization. I describe here the process I followed to deploy this kind of algorithm in production. **I would like to point out that we all had different experiences with deploying algorithms in production.** You may not agree with all of the points I'm making. I invite you to share your view in the comments. 

**I distinguish two main steps in the organization of the project when deploying a machine learning algorithm in production:**

- The **prototyping phase** phase is dedicated to the selection of the algorithm and the selection of the features to solve the problem.


- The **engineering phase** phase is dedicated to the creation of robust machine learning systems. It aims at deploying the model in production, allows re-training the model on a regular basis.


#### Prototyping:

The first thing to do during the prototyping phase phase is to define a method for evaluating the quality of the model. **Which objective do you want to optimize?** You have to define a validation process now. Usually this is cross-validation. After defining the validation process, the whole point is to find the most suitable model with carefully selected hyperparameters. Without forgetting the feature engineering stage, which is the key to most problems. 

The prototyping step is difficult and exciting. We rely on our expertise in the field concerned, our creativity and our scientific culture.

#### Engineering:

It seems interesting to me to choose to deploy the algorithm for predicting product sales behind an API. The API is a practical solution to allow the largest number of users and software to query the trained model. 

During the engineering phase I distinguish two sub-categories modules. The first one is dedicated to the training of the model and its serialization. I call the first set **Offline**. The second one is dedicated to the behavior of the model in the production environment. I call this second part **Online**. I call it "online" because, in my opinion, deploying the model behind an API is an interesting solution here. 

There is a lot of engineering work to ensure consistency between the offline training part and the online inference part. Any transformations that have been applied to the data during training must be applied to the data during the inference phase. This requires the development of code that is different from the training phase, but which produces the same results.


The development phase should lead to the creation of different modules:

**Offline:**

- **module 1:** Script dedicated to the calculation of features for the model training. The feature computation should be vectorized to speed up the process.


- **module 2:** Script for training, and evaluating the model. The training of the model is based on the features computed by the module 1.


- **module 3:** Script dedicated to the serialization of the model. It is important to redefine the model prediction method before serializing the model. Libraries like Scikit-Learn do not develop models so that they can quickly make predictions for a single observation. You can find more information [here](https://maxhalford.github.io/blog/speeding-up-sklearn-single-predictions/). The [sklearn-onnx](https://github.com/onnx/sklearn-onnx) library is an interesting solution to this problem. I already used [treelite](https://github.com/dmlc/treelite) and this is a suitable alternative for LightGBM.

![](static/offline.png)

**Online:**

- **module 4**: Script for calculating features for the production environment. Usually the predictions in the production environment are made via the call of an API. The feature calculation should not be vectorized because it is performed for a single observation when calling the API. As a result, the source code of module 4 differs from the source code of module 1.

**Deployment:**

- **Module 5**: API definition. When a call is made, the API must load the serialized model, calculate the features using module 4 and make a prediction. The model could also be loaded into memory at API startup.

![](static/online.png)

**Tests:**

- It is strongly recommended to integrate multiple unit tests such as unit tests for offline feature computation and unit test for online feature calculation. Non regression test to ensure that the offline model produces the same results as the online model are necessary too.

**After deploying an algorithm in production, you will need to re-train the model regularly and maintain the architecture. Deploying a learning machine algorithm that learns by batch is tedious. It's a long-term project that requires a lot of rigor. Such a project represents a significant technological debt and monopolizes highly qualified human resources on a daily basis.**

### Model deployment with Creme and Chantilly

Creme is an online learning library. Creme allows to train machine learning models on data flows. 

Each Creme model has a ``fit_one`` method. **The ``fit_one`` method allows to update the model when there is a new observation available** for training. Similar to neural networks, there is no need to re-train the model from scratch when new observations come in.

Creme is not a suitable solution for Kaggle. Learning in batch allows the model to converge faster. **I won't choose Creme to get a medal on Kaggle. However, in everyday life, Creme is a viable and flexible solution for modeling a complex problem**.

In this kernel, I am going to make a tutorial to show how to deploy in production a Creme algorithm trained to predict the target of the M5-Forecasting-Accuracy competition. I'll use the library [Chantilly](https://github.com/creme-ml/chantilly) to deploy my solution in production. Chantilly is a library under development that allows you to easily deploy the models from Creme in production.

#### Prototyping

As usual, during the prototyping phase, I define the validation process and the metrics used to evaluate the quality of the model I develop. Online learning allows to do **progressive validation** which is the online counterpart of cross-validation. The progressive validation allows to take into account the temporality of the problem. For reasons of simplicity, I choose to use the MAE metric to evaluate the quality of my model.

After a few tries on my side, **I choose to train a ``KNNRegressor``  and a ``LinearRegression`` per product** to predict the number of sales. It represents **30490 * 2 models** models. I will choose the best of the two models for each of the products thanks to the validation score. 

#### Engineering

I chose to use the development version of Creme because there is a feature (``Shift``) that I use that is not yet in the public version of Creme. 

```bash
pip install git+https://github.com/creme-ml/creme --upgrade
```

I'm importing the packages that I need to train my models

```python
import copy
import collections 
import datetime
import random
import tqdm
```

```python
from creme import compose
from creme import feature_extraction
from creme import linear_model
from creme import metrics
from creme import neighbors
from creme import optim
from creme import preprocessing
from creme import stats
from creme import stream
```

I use this first function to parse the date and extract the number of the day.

```python
def extract_date(x):
    """Extract features from the date."""
    import datetime
    if not isinstance(x['date'], datetime.datetime):
        x['date'] = datetime.datetime.strptime(x['date'], '%Y-%m-%d')
    x['wday'] = x['date'].weekday()
    return x
```

``get_metadata`` allow me to extract the identifier of the product.

```python
def get_metadata(x):
    key = x['id'].split('_')
    x['item_id'] = f'{key[0]}_{key[1]}_{key[2]}'
    return x
```

Below I define the feature extraction pipeline. I use the module ``feature_extraction.TargetAgg`` to calculate the features on the target variable of the stream. I calculate rolling features on the target variable. I use the ``Shift`` module to easily compute features with a lag. I calculate rolling averages on the target variable with a lag of about 30 days. I also calculate the rolling average of the target variable as a function of the day.

```python
extract_features = compose.TransformerUnion(
    compose.Select('wday'),
    
    feature_extraction.TargetAgg(by=['item_id'], how=stats.Mean()),
    feature_extraction.TargetAgg(by=['item_id'], how=stats.Var()),

    feature_extraction.TargetAgg(by=['item_id'], how=stats.RollingMean(1)),
    feature_extraction.TargetAgg(by=['item_id'], how=stats.RollingMean(3)),
    feature_extraction.TargetAgg(by=['item_id'], how=stats.RollingMean(7)),
    feature_extraction.TargetAgg(by=['item_id'], how=stats.RollingMean(15)),
    feature_extraction.TargetAgg(by=['item_id'], how=stats.RollingMean(20)),
    feature_extraction.TargetAgg(by=['item_id'], how=stats.RollingMean(30)),
    
    feature_extraction.TargetAgg(by=['item_id'], how=stats.Shift(30) | stats.RollingMean(1)),
    feature_extraction.TargetAgg(by=['item_id'], how=stats.Shift(29) | stats.RollingMean(1)),
    feature_extraction.TargetAgg(by=['item_id'], how=stats.Shift(28) | stats.RollingMean(1)),
    feature_extraction.TargetAgg(by=['item_id'], how=stats.Shift(27) | stats.RollingMean(1)),
    feature_extraction.TargetAgg(by=['item_id'], how=stats.Shift(26) | stats.RollingMean(1)),
    
    feature_extraction.TargetAgg(by=['wday'], how=stats.RollingMean(1)),
    feature_extraction.TargetAgg(by=['wday'], how=stats.RollingMean(3)),
    feature_extraction.TargetAgg(by=['wday'], how=stats.RollingMean(7)),
    feature_extraction.TargetAgg(by=['wday'], how=stats.RollingMean(15)),
    feature_extraction.TargetAgg(by=['wday'], how=stats.RollingMean(20)),
    feature_extraction.TargetAgg(by=['wday'], how=stats.RollingMean(30)),
)
```

I will train two models per product, which represents **30490 * 2 models**. The first model with every product is a ``KNeighborsRegressor``. The second is a linear model. I noticed that these two models are complementary. I will select the best of the two models for each product as the model I will deploy in production.

The code below allows me to declare my two pipelines, the first one dedicated to KNN and the second one to the linear model.

```python
# Init pipeline dedicated to KNN
knn = (
    compose.FuncTransformer(get_metadata) |
    compose.FuncTransformer(extract_date) |
    extract_features |
    neighbors.KNeighborsRegressor(window_size=300, n_neighbors=30, p=2)
)


# Init pipeline dedicated to linear model
lm = (
    compose.FuncTransformer(get_metadata) |
    compose.FuncTransformer(extract_date) |
    extract_features |
    linear_model.LinearRegression(optimizer=optim.SGD(0.00005), clip_gradient=1, intercept_lr=0.001)
)
```

The piece of code below creates a copy of both pipelines for all products and store them in a dictionary.

```python
list_model = []

X_y = stream.iter_csv('./data/sample_submission.csv', target_name='F8')

for x, y in tqdm.tqdm(X_y, position=0):
    
    item_id = '_'.join(x['id'].split('_')[:5])

    if item_id not in list_model:

        list_model.append(item_id)
        
dict_knn = {item_id: copy.deepcopy(knn) for item_id in tqdm.tqdm(list_model, position=0)}
dict_lm  = {item_id: copy.deepcopy(lm) for item_id in tqdm.tqdm(list_model, position=0)}
```

I do a warm-up of all the models from a subset of the training set. To do this pre-training, I selected the last two months of the training set and saved it in csv format. I use Creme's ``stream.iter_csv`` module to iterate on the training dataset. The pipeline below consumes very little RAM memory because we load the data into the memory one after the other.

```python
random.seed(42)

params = dict(target_name='y', converters={'y': int, 'id': str}, parse_dates= {'date': '%Y-%m-%d'})

# Init streaming csv reader
X_y = stream.iter_csv('./data/train_preprocessed.csv', **params)

bar = tqdm.tqdm(X_y, position = 0)

# Init online metrics:
metric_knn = collections.defaultdict(lambda: metrics.MAE())
metric_lm  = collections.defaultdict(lambda: metrics.MAE())

for i, (x, y) in enumerate(bar):
    
    # Extract item id
    item_id  = '_'.join(x['id'].split('_')[:5])
    
    # KNN
    
    # Evaluate performance of KNN
    y_pred_knn = dict_knn[f'{item_id}'].predict_one(x)
    
    # Update metric of KNN
    metric_knn[f'{item_id}'].update(y, y_pred_knn)
    
    # Fit KNN
    dict_knn[f'{item_id}'].fit_one(x=x, y=y)
    
    # Linear Model
    
    # Evaluate performance of linear model
    y_pred_lm  = dict_lm[f'{item_id}'].predict_one(x)
    
    # Update metric of linear model
    metric_lm[f'{item_id}'].update(y, y_pred_lm)
    
    # Train linear model for 10 epochs on each training example
    for _ in range(10):
        dict_lm[f'{item_id}'].fit_one(x=x, y=y)
```

I select the best model among the knn and the linear model for the 30490 products and save my models:

```python
models = {}

for item_id in tqdm.tqdm(scores_knn.keys()):
    
    score_knn = scores_knn[item_id]
    
    score_lm  = scores_lm[item_id]
    
    if score_knn < score_lm:
        models[item_id] = dict_knn[item_id]
        
    else:
        models[item_id] = dict_lm[item_id]
        
# Save selected models:
with open('models.dill', 'wb') as file:
    dill.dump(models, file)
```


#### Deployment of the model:

**Now that all the models are pre-trained, I will be able to deploy the pipelines behind an API in a production environment. I will use the [Chantilly](https://github.com/creme-ml/chantilly) library to do so.**

**[Chantilly](https://github.com/creme-ml/chantilly) is a project that aims to ease train Creme models when they are deployed. Chantilly is a minimalist API based on the Flask framework.** Chantilly allows to make predictions, train models and measure model performance in real time. It gives access to a dashboard.

Chantilly is a library currently under development. For various reasons, I choose to extract the files from Chantilly that I'm interested in to realize this project. I chose to deploy my API with [Digital Ocean](https://www.digitalocean.com). You will be able to find the whole architecture of my API [here](https://github.com/raphaelsty/M5-Forecasting-Accuracy). 


To deploy my API, I followed the following steps:


- I selected the server on Digital Ocean with the smallest configuration


- Tutorial to initialize my server and firewall [here](https://www.digitalocean.com/community/tutorials/initial-server-setup-with-ubuntu-16-04)


- Tutorial to install Anaconda on my server [here](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04)


- Allow reading on port 8080 to be able to request my API ``sudo ufw allow 8080``


- Installation of git ``sudo apt install git``


- Clone my API on the server ``git clones https://github.com/raphaelsty/M5-Forecasting-Accuracy.git``


- I installed the requirements of my repo ``pip install -r requirements.txt``.


- I went to the repository I cloned and ran the following command to start my API:
``waitress-serve --call 'app:create_app'``.

That's it.

I initialize my API with flavor regression (see Chantilly tutorial):

```python
import requests
url = 'http://159.89.191.92:8080'
```

```python
requests.post(f'{url}/api/init', json= {'flavor': 'regression'})
```

After initializing the flavor of my API, I upload all the models I've pre-trained. Each model has a name. This name is the name of the product. I have used dill to serialize the model before uploading it to my API.

```python
for model_name, model in tqdm.tqdm(models.items(), position=0):
    r = requests.post(f'{url}/api/model/{model_name}', data=dill.dumps(model))
```

All the models are now deployed in production and available to make predictions. The models can also be updated on a daily basis. That's it.

![](static/online_learning.png)

**As you may have noticed, the philosophy of online learning allows to reduce the complexity of the deployment of a machine learning algorithm in production. Moreover, to update the model, we only have to make calls to the API. We don't need to re-train the model from scratch.** To maintain my models on a daily basis, I recommend setting up a script that queries the database that stores the sales made that day. This script would perform 30490 queries every day to update all the models.

#### Make a prediction by calling the API:

```python
r = requests.post(f'{url}/api/predict', json={
    'id': 1,
    'model': 'HOBBIES_1_001_CA_1',
    'features': {'date': '2016-05-23', 'id': 'HOBBIES_1_001_CA_1'}
})
```

#### Update models with new data:

```python
r = requests.post(f'{url}/api/learn', json={
    'id': 1,
    'model': 'HOBBIES_1_001_CA_1',
    'ground_truth': 1,
})
```

#### Chantilly dashboard

You can consult the [dashboard](http://159.89.191.92:8080) which is updated in real time and know the performance of your models in live.

![](static/dashboard.png)

Feel free to visit the [Chantilly](https://github.com/creme-ml/chantilly) github for more details on the API features.

--

Thank you for reading me. 

Raphaël Sourty.
