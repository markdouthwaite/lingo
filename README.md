# lingo

A package for quickly deploying Scikit-Learn Linear Models as Go applications. 

It was developed to explore use-cases requiring fast inference for very large numbers of models, and to do this
 cost-effectively. 

The package has been tested with the following Scikit-Learn Linear Model variants:

* **LinearRegression**
* **LogisticRegression**
* **Ridge**
* **RidgeClassifier**
* **Lasso**
* **SGDRegressor**
* **SGDClassifier**

While the package is designed for some of the most common Scikit-Learn linear models, **it was designed to demonstrate 
a concept that can easily be applied to other model variants too** -- and not just models in Scikit-Learn.

## Quickstart

Before you begin, you'll need to save your Scikit-Learn model with the [`py-lingo` Python package](https://pypi.org/project/py-lingo/). When you've got that
saved, you can move to deploy your model...

If you're familiar with the setting up servers in Go, then this should all feel familiar. All you need to do is:

1. Load either your `LinearClassifier` or `LinearRegressor` model with `LoadClassifier` or `LoadRegressor` functions respectively.
2. Create a handler for your model with the `NewRegressorHandler` or `NewClassifierHandler` function.
3. Setup your server as normal, passing your newly created handler to whichever router you choose!

Here's a code example of the above for one of the sample models:  

```go
package main 

import (
    "github.com/markdouthwaite/lingo"
    "github.com/gorilla/mux"
	"net/http" 
    "time"
	"log"
)


func main(){
    model := lingo.LoadRegressor("artifacts/boston.h5")
    handler := lingo.NewRegressorHandler(model)

    router := mux.NewRouter()
    router.HandleFunc("/predict", handler)

	server := &http.Server{
		Handler:      router,
		Addr:         "127.0.0.1:8000",
		// Good practice: enforce timeouts for servers you create!
		WriteTimeout: 15 * time.Second,
		ReadTimeout:  15 * time.Second,
	}

	log.Fatal(server.ListenAndServe())

}
```

Tailored to run inference on individual feature vectors (i.e. single observations).

## Bundled models

There's a few models bundled with this package for testing purposes. They are:

* `data/classifiers/iris.h5` - A Scikit-Learn LogisticRegression model trained on the Iris plant dataset.
* `data/regressors/boston.h5` - A Scikit-Learn LinearRegression model trained on a reduced version of the Boston housing dataset. 
* `data/regressors/multi-boston.h5` - A Scikit-Learn LinearRegression model trained on a reduced version of the Boston housing dataset, but with two output variables (both identical). 
