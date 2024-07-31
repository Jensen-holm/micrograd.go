package main

import (
	microgradgo "Jensen-holm/micrograd.go/micrograd"
	"math"
	"testing"
)

func TestNewValue(t *testing.T) {
	val := microgradgo.NewValue(8.0)
	if val.Data != 8.0 {
		t.Fatalf("TestNewValue failed ::: %g != 8.0", val.Data)
	}
}

func TestAdd(t *testing.T) {
	val1 := microgradgo.NewValue(8.0)
	val2 := microgradgo.NewValue(10.0)
	result := val1.Add(val2)
	exp := val1.Data + val2.Data
	if result.Data != exp {
		t.Fatalf("TestAdd failed ::: %g + %g != %g", val1.Data, val2.Data, exp)
	}
}

func TestMul(t *testing.T) {
	val1 := microgradgo.NewValue(100.0)
	val2 := microgradgo.NewValue(10.0)
	result := val1.Mul(val2)
	exp := val1.Data * val2.Data
	if result.Data != exp {
		t.Fatalf("TestMul failed ::: %g * %g != %g", val1.Data, val2.Data, exp)
	}
}

func TestPow(t *testing.T) {
	val1 := microgradgo.NewValue(1000)
	val2 := microgradgo.NewValue(7)
	result := val1.Pow(val2)
	exp := math.Pow(val1.Data, val2.Data)
	if result.Data != exp {
		t.Fatalf("TestPow failed ::: %g ** %g != %g", val1.Data, val2.Data, exp)
	}
}

func TestNewValueChildren(t *testing.T) {
	val1 := microgradgo.NewValue(10)
	val2 := microgradgo.NewValue(8.0)
	result := val1.Add(val2)

	_, val1Ok := result.Children[val1]
	_, val2Ok := result.Children[val2]
	if !val2Ok || !val1Ok {
		t.Fatalf("TestNewValueChildren failed ::: children were not added properly")
	}
}
