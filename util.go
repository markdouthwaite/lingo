package lingo

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/hdf5"
	"math"
)

const (
	// ErrorThreshold is used in tests as the default error expected between predicted and expected values.
	ErrorThreshold = 0.00001
)

type Validator interface {
	validate(x []float64) error
}

// VecToArrayFloat64 converts a vector into an array of floats.
func VecToArrayFloat64(v mat.Vector) []float64 {
	r, _ := v.Dims()
	o := make([]float64, r)
	for i := 0; i < r; i++ {
		o[i] = v.At(i, 0)
	}
	return o
}

// Argmax gets the i, j associated with the largest value in a dense matrix.
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

// loadArray loads a 1D float array from a HDF5 dataset.
func loadArray(dataset *hdf5.Dataset) ([]float64, int, int) {
	rows, cols := 0, 0
	rowsAttr, err := dataset.OpenAttribute("n")

	if err != nil {
		panic("failed to open attribute 'n'")
	}

	colsAttr, err := dataset.OpenAttribute("m")

	if err != nil {
		panic("failed to open attribute 'n'")
	}

	err = colsAttr.Read(&rows, hdf5.T_NATIVE_UINT32)

	if err != nil {
		panic("failed to read 'rows'")
	}

	err = rowsAttr.Read(&cols, hdf5.T_NATIVE_UINT32)

	if err != nil {
		panic("failed to read 'cols'")
	}

	params := make([]float64, rows*cols)

	err = dataset.Read(&params)

	return params, rows, cols
}

// Load loads a linear model from a HDF5 file.
func Load(fileName string) (modelType string, model *LinearModel) {

	file, err := hdf5.OpenFile(fileName, hdf5.F_ACC_RDONLY)
	if err != nil {
		panic("Failed to open target HDF5 file: " + fileName)
	}

	defer file.Close()

	modelGroup, err := file.OpenGroup("model")

	if err != nil {
		panic("failed to open group 'model'")
	}

	modelTypeAttr, err := modelGroup.OpenAttribute("estimatorType")

	if err != nil {
		panic("failed to open attribute 'estimatorType'")
	}

	modelTypeAttr.Read(&modelType, hdf5.T_GO_STRING)

	thetaDataset, err := modelGroup.OpenDataset("theta")

	defer thetaDataset.Close()

	if err != nil {
		panic("failed to open dataset 'theta'")
	}

	interceptDataset, err := modelGroup.OpenDataset("intercept")

	defer interceptDataset.Close()

	if err != nil {
		panic("failed to open dataset 'intercept'")
	}

	theta, _, nVars := loadArray(thetaDataset)
	intercept, _, _ := loadArray(interceptDataset)

	model = NewLinearModel(theta, intercept, nVars)

	return

}

func LoadClassifier(fileName string) (model *LinearClassifier) {
	modelType, coreModel := Load(fileName)

	if modelType != "classifier" {
		panic("expected model of type 'classifier', got: " + modelType)
	}

	model = &LinearClassifier{coreModel}

	return
}
