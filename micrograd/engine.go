package microgradgo

import (
	"fmt"
	"math"
)

const (
	NIL  = ""
	ADD  = "+"
	MUL  = "*"
	POW  = "^"
	TANH = "tanh"
	RELU = "relu"
)

type Value struct {
	data         float64
	prev         map[*Value]bool
	op           string
	grad         float64
	backwardFunc func()
}

// NewValue: for constructing Values only that are not
// a part of any computational graph yet
func NewValue(data float64) *Value {
	return &Value{
		data:         data,
		prev:         nil,
		op:           NIL,
		backwardFunc: nil,
		grad:         0,
	}
}

// opResult: for constructing values from within an operation
func opResult(data float64, prev map[*Value]bool, op string) *Value {
	return &Value{
		data:         data,
		prev:         prev,
		op:           op,
		backwardFunc: nil,
		grad:         0,
	}
}

// newprev: used within operators so that I don't have to
// write a map literal for every single operation
func newprev(v, other *Value) map[*Value]bool {
	return map[*Value]bool{
		v:     true,
		other: true,
	}
}

// --- OPERATORS --- //

func (v *Value) Add(other *Value) *Value {
	result := opResult(v.data+other.data, newprev(v, other), ADD)
	result.backwardFunc = func() {
		v.grad += result.grad
		other.grad += result.grad
	}
	return result
}

func (v *Value) Mul(other *Value) *Value {
	result := opResult(v.data*other.data, newprev(v, other), MUL)
	result.backwardFunc = func() {
		v.grad += other.data * result.grad
		other.grad += v.grad * result.grad
	}
	return result
}

func (v *Value) Pow(other *Value) *Value {
	result := opResult(math.Pow(v.data, other.data), newprev(v, other), POW)
	result.backwardFunc = func() {
		v.grad += (other.data * math.Pow(v.data, other.data-1)) * result.grad
	}
	return result
}

func (v *Value) Tanh() *Value {
	t := (math.Exp(2*v.data) - 1) / (math.Exp(2 * v.data))
	result := opResult(t, newprev(v, nil), TANH)
	result.backwardFunc = func() {
		v.grad += (1 - t*t) * result.grad
	}
	return result
}

func (v *Value) Relu() *Value {
	result := opResult(max(v.data, 0), newprev(v, nil), RELU)
	result.backwardFunc = func() {
		if result.data > 0 {
			v.grad += result.grad
		}
	}
	return result
}

// --- MAIN BAKCPROP METHOD & HELPERS --- //

func buildTopoOrder(v *Value, visited map[*Value]bool, topo []*Value) []*Value {
	if _, vIsVisited := visited[v]; vIsVisited {
		return topo
	}

	visited[v] = true
	for child := range v.prev {
		topo = buildTopoOrder(child, visited, topo)
	}
	topo = append(topo, v)
	return topo
}

func (v *Value) Backward() {
	topo := make([]*Value, 0)
	visited := make(map[*Value]bool, 0)
	topo = buildTopoOrder(v, visited, topo)

	v.grad = 1
	for idx := len(topo) - 1; idx >= 0; idx-- {
		val := topo[idx]
		val.backwardFunc()
		fmt.Println(val)
	}
}
