# lingo

A package for quickly deploying Scikit-Learn Linear Models as Go applications.

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
    model := lingo.Load("artifacts/model.h5")
    router := mux.NewRouter()
    router.HandleFunc("/predict", NewModelHandler(model))

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
