**Hi kagglers**,

Deploying and maintaining machine learning algorithms in a **production** environment is not an easy task. The **drift** of data over the time tends to degrade the performance of the algorithms because the models are static. Data Scientist **re-train models from scratch** to update them. This task is tedious and monopolizes highly qualified human resources. 

**I would like to present a solution to these problems**. I will use online learning and the open-source **[Creme](https://github.com/creme)** library (I am a core developer of Creme) to overcome the difficulties of deploying a machine learning model in production. I will illustrate my point with data from the **[M5-Forecasting-Accuracy kaggle competition](https://www.kaggle.com/c/m5-forecasting-accuracy/)** which is well suited to the use case of Creme. 

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

After a few tries on my side, **I choose to train a ``KNNRegressor`` model per product** to predict the number of sales. It represents 3049 models. All the models provide correct results with the progressive validation method. I train my models to predict sales 7 days in advance.

I choose to use as features for each model:

- Global mean per store.

- Global standard deviation per store.

- Average sales for the last 1, 3, 7, 15, 30 days per store.

- Average of the sales per score according to the day, ie {Monday, Tuesday, ..Sunday} with a lag of 1, 3, 7, 15, 30 days.

#### Engineering

I'll start by installing the Creme library:

```bash
pip install creme
```

I'm importing the packages that I'm going to need:

```python
import copy
import datetime
import random
import tqdm
```

```python
from creme import compose
from creme import feature_extraction
from creme import neighbors
from creme import metrics
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

``get_metadata`` allows you to extract the identifier of the product and the store.

```python
def get_metadata(x):
    key = x['id'].split('_')
    x['store_id'] = f'{key[3]}_{key[4]}'
    x['item_id'] = f'{key[0]}_{key[1]}_{key[2]}'
    return x
```

Below I define the feature extraction pipeline. I use the module ``feature_extraction.TargetAgg`` to calculate the features on the target variable of the stream.

```python
extract_features = compose.TransformerUnion(
    compose.Select('wday'),
    
    feature_extraction.TargetAgg(by=['store_id'], how=stats.Mean()),
    feature_extraction.TargetAgg(by=['store_id'], how=stats.Var()),
    
    feature_extraction.TargetAgg(by=['store_id', 'wday'], how=stats.RollingMean(30)),
    feature_extraction.TargetAgg(by=['store_id', 'wday'], how=stats.RollingMean(15)),
    feature_extraction.TargetAgg(by=['store_id', 'wday'], how=stats.RollingMean(7)),
    feature_extraction.TargetAgg(by=['store_id', 'wday'], how=stats.RollingMean(3)),
    feature_extraction.TargetAgg(by=['store_id', 'wday'], how=stats.RollingMean(1)),
    
    feature_extraction.TargetAgg(by=['store_id'], how=stats.RollingMean(1)),
    feature_extraction.TargetAgg(by=['store_id'], how=stats.RollingMean(30)),
    feature_extraction.TargetAgg(by=['store_id'], how=stats.RollingMean(15)),
    feature_extraction.TargetAgg(by=['store_id'], how=stats.RollingMean(7)),
    feature_extraction.TargetAgg(by=['store_id'], how=stats.RollingMean(3)),
)
```

Below, I define the global pipeline I want to deploy in production. The pipeline is composed of:

- Extraction of the product identifier.

- Extraction of the day number of the date $\in$ {1, 2, ..7}. 

- Computation of the features.

- Standard scaler that centers and reduces the value of features.

- Model declaration ``neighbors.KNeighborsRegressor``.

```python
model = (
    compose.FuncTransformer(get_metadata) |
    compose.FuncTransformer(extract_date) |
    extract_features |
    preprocessing.StandardScaler() |
    neighbors.KNeighborsRegressor(window_size=30, n_neighbors=15)
)
```

I have choosen to create one model per product. The piece of code below creates a copy of the pipeline for all products and store them in a dictionary.

```python
list_model = []

X_y = stream.iter_csv('./data/sample_submission.csv', target_name='F8')

for x, y in tqdm.tqdm(X_y, position=0):
    
    item_id = '_'.join(x['id'].split('_')[:5])
    
    if item_id not in list_model:
    
        list_model.append(item_id)
        
dic_models = {item_id: copy.deepcopy(model) for item_id in tqdm.tqdm(list_model, position=0)}
```

I do a warm-up of all the models from a subset of the training set. To do this pre-training, I selected the last two months of the training set and saved it in csv format.I use Creme's ``stream.iter_csv`` module to iterate on the training dataset. The pipeline below consumes very little RAM memory because we load the data into the memory one after the other. I train my models to predict sales 7 days in advance.

```python
random.seed(42)

params = dict(target_name='y', converters={'y': int, 'id': str}, parse_dates= {'date': '%Y-%m-%d'})

X_y = stream.simulate_qa(
    X_y    = stream.iter_csv('./data/train.csv', **params), 
    moment = 'date', 
    delay  = datetime.timedelta(days=7)
)

bar = tqdm.tqdm(X_y, position = 0)

metric = metrics.Rolling(metrics.MAE(), 30490)

y_pred = {}

for i, x, y in bar:
    
    item_id  = '_'.join(x['id'].split('_')[:3])
    store_id = '_'.join(x['id'].split('_')[3:5])
    
    if y:
        dic_models[f'{item_id}'].fit_one(x=x, y=y)
        
        # Update the metric:
        metric = metric.update(y, y_pred[f'{item_id}_{store_id}'])
        
        if i % 1000 == 0:
            # Update tqdm progress bar.
            bar.set_description(f'MAE: {metric.get():4f}')

    else:

        y_pred[f'{item_id}_{store_id}'] = dic_models[f'{item_id}'].predict_one(x)
```

#### Deployment of the model:

**Now that all the models are pre-trained, I will be able to deploy the pipelines behind an API in a production environment. I will use the [Chantilly](https://github.com/creme-ml/chantilly) library to do so.**

**[Chantilly](https://github.com/creme-ml/chantilly) is a project that aims to ease train Creme models when they are deployed. Chantilly is a minimalist API based on the Flask framework.** Chantilly allows to make predictions, train models and measure model performance in real time. It gives access to a dashboard.

Chantilly is a library currently under development. For various reasons, I choose to extract the files from Chantilly that I'm interested in to realize this project.

I choose to deploy my API on Heroku. To do so I followed the [tutorial](https://stackabuse.com/deploying-a-flask-application-to-heroku/). I choose Heroku because they allow me to run my API with a very modest configuration at a low cost. (This modest configuration increases the response time of my API when there are several users).

The main difficulty I encountered when deploying on Heroku was creating the ``Profile`` file. The ``Procfile`` is used to initialize the API when it is deployed on Heroku.

Here is its contents:

```web: gunicorn -w 4 "app:create_app()"```

You will be able to find the whole architecture of my API [here](https://github.com/raphaelsty/M5-Forecasting-Accuracy).

After deploying my Chantilly API on Heroku, I add the regression flavor. Chantilly uses this flavor to select the appropriate metrics (MAE, MSE and SMAPE).

```python
import requests

r = requests.post('http://localhost:5000/api/init', json= {'flavor': 'regression'})
```

After initializing the flavor of my API, I upload all the models I've pre-trained. Each model has a name. This name is the name of the product. I have used dill to serialize the model before uploading it to my API.

```python
import dill

for model_name, model in dic_models.items():
    
    r = requests.post('http://localhost:5000/api/model/{}'.format(model_name), data=dill.dumps(model))
    
```

All the models are now deployed in production and available to make predictions. The models can also be updated on a daily basis. That's it.

![](static/online_learning.png)

**As you may have noticed, the philosophy of online learning allows to reduce the complexity of the deployment of a machine learning algorithm in production. Moreover, to update the model, we only have to make calls to the API. We don't need to re-train the model from scratch.**

#### Make a prediction by calling the API:

```python
json = {
    'id': 1,
    'model': 'HOBBIES_1_001',
    'features': {'date': '2020-04-30', 'id': 'HOBBIES_1_001_CA_1'}
}

r = requests.post('http://localhost:5000/api/predict', json=json)

prediction = r.json()
```

#### Update models with new data:

```python
r = requests.post('http://localhost:5000/api/learn', json={
    'id': 1,
    'model': 'HOBBIES_1_001',
    'ground_truth': 1,
})
```
You can execute these requests. I don't recommend that you request my models to blend my predictions to yours. My models are not intended to be competitive. 

If you want my opinion on the competition, I think there's a lot of noise in the data. I haven't observed any particular trends when it comes to day-to-day predictions for all products.


Feel free to visit the [Chantilly](https://github.com/creme-ml/chantilly) github for more details on the API features.

--

Thank you for reading me. 

Raphaël Sourty.
