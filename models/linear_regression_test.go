package models

import (
	"fmt"
	"testing"
)


func setupLinearRegression() *LinearRegression{
	coeffs := []float64{0.3574458,  -0.01453134, -1.2603342,  0.00891642, -0.79181815}
	intercept := 55.15045304131303
	return CreateLinearRegression(coeffs, intercept, 1)
}


func setupMultipleLinearRegression(){

}


func TestLinearRegression_Predict(t *testing.T) {
	model := setupLinearRegression()
	fmt.Println(model.coeffs)
}
