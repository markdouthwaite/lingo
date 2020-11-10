package lingo

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

const (
	ErrorThreshold = 0.00001
)

// Convert a vector into an array of floats.
func VecToArrayFloat64(v mat.Vector) []float64 {
	r, _ := v.Dims()
	o := make([]float64, r)
	for i := 0; i < r; i++ {
		o[i] = v.At(i, 0)
	}
	return o
}

// Get the i, j associated with the largest value in a dense matrix.
func Argmax(a mat.Vector) (int, int) {
	r, c := a.Dims()
	maxVal := math.Inf(-1)
	maxI := 0
	maxJ := 0
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			val := a.At(i, j)
			if val > maxVal {
				maxVal = val
				maxI = i
				maxJ = j
			}
		}
	}
	return maxI, maxJ
}
