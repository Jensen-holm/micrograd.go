package microgradgo

import (
	"math"
	"testing"
)

func TestNewValue(t *testing.T) {
	val := NewValue(8.0)
	if val.data != 8.0 {
		t.Fatalf("TestNewValue failed ::: %g != 8.0", val.data)
	}
	if val.op != NIL {
		t.Fatalf("TestNewValue failed ::: wrong op assigned")
	}
}

func TestAdd(t *testing.T) {
	val1 := NewValue(8.0)
	val2 := NewValue(10.0)
	result := val1.Add(val2)
	exp := val1.data + val2.data
	if result.data != exp {
		t.Fatalf("TestAdd failed ::: %g + %g != %g", val1.data, val2.data, exp)
	}
	if result.op != ADD {
		t.Fatalf("TestAdd failed ::: wrong op assigned to result")
	}
	if result.backwardFunc == nil {
		t.Fatalf("TestAdd failed ::: backwards function was not set in result after operation")
	}
}

func TestMul(t *testing.T) {
	val1 := NewValue(100.0)
	val2 := NewValue(10.0)
	result := val1.Mul(val2)
	exp := val1.data * val2.data
	if result.data != exp {
		t.Fatalf("TestMul failed ::: %g * %g != %g", val1.data, val2.data, exp)
	}
	if result.op != MUL {
		t.Fatalf("TestMul failed ::: wrong op assigned to result")
	}
	if result.backwardFunc == nil {
		t.Fatalf("TestMul failed ::: backwards function was not set in result after operation")
	}
}

func TestPow(t *testing.T) {
	val1 := NewValue(1000)
	val2 := NewValue(7)
	result := val1.Pow(val2)
	exp := math.Pow(val1.data, val2.data)
	if result.data != exp {
		t.Fatalf("TestPow failed ::: %g ** %g != %g", val1.data, val2.data, exp)
	}
	if result.op != POW {
		t.Fatalf("TestPow failed ::: wrong op assigned to result")
	}
	if result.backwardFunc == nil {
		t.Fatalf("TestPow failed ::: backwards function was not set in result after operation")
	}
}

func TestReluNeg(t *testing.T) {
	val1 := NewValue(-1)
	result := val1.Relu()
	if result.data != 0 {
		t.Fatalf("TestReluNeg failed ::: relu(%g) != 0", val1.data)
	}
	if result.op != RELU {
		t.Fatalf("TestReluNeg failed ::: wrong op assigned to result")
	}
	if result.backwardFunc == nil {
		t.Fatalf("TestReluNeg failed ::: backwards function was not set in result after operation")
	}
}

func TestReluPos(t *testing.T) {
	val1 := NewValue(1)
	result := val1.Relu()
	if result.data != val1.data {
		t.Fatalf("TestReluPos failed ::: relu(%g) != %g", val1.data, val1.data)
	}
	if result.op != RELU {
		t.Fatalf("TestReluPos failed ::: wrong op assigned to result")
	}
	if result.backwardFunc == nil {
		t.Fatalf("TestReluPos failed ::: backwards function was not set in result after operation")
	}
}

func TestNewValueprev(t *testing.T) {
	val1 := NewValue(10)
	val2 := NewValue(8.0)
	result := val1.Add(val2)

	_, val1Ok := result.prev[val1]
	_, val2Ok := result.prev[val2]
	if !val2Ok || !val1Ok {
		t.Fatalf("TestNewValueprev failed ::: prev were not added properly")
	}
}

// --- TESTING BACKPROP --- //

func TestBuildTopo(t *testing.T) {
	// ((4 * 5) + 4) * 5 = 120
	a := NewValue(4)
	b := NewValue(5)
	result := a.Mul(b).Add(a).Mul(b)
	if result.data != 120 {
		t.Fatalf("TestBuildTopo failed ::: (4 * 5 + 4) * 5 should equal 120. not '%g'", result.data)
	}

	visited := map[*Value]bool{}
	topo := []*Value{}
	topo = buildTopoOrder(result, visited, topo)
	if len(topo) != 5 {
		/// the length of topo should be the number of operations performed + 1
		t.Fatalf("TestBuildTopo failed ::: length of topo is '%d' when it should be 5 ::: topo = %#v", len(topo), topo)
	}
}

func TestAddBackward(t *testing.T) {
	a := NewValue(1)
	b := NewValue(2)
	result := a.Add(b)
	result.Backward()
	if a.grad != 1.0 {
		t.Fatalf("TestAddBackward failed ::: gradient of a in 'a + b' expected to be 1.0, not %g", a.grad)
	}
	if b.grad != 1.0 {
		t.Fatalf("TestAddBackward failed ::: gradient of b in 'a + b' expected to be 1.0, not %g", b.grad)
	}
}
func TestMulBackward(t *testing.T) {
	a := NewValue(4)
	b := NewValue(5)
	result := a.Mul(b)
	result.Backward()
	if a.grad != b.data {
		t.Fatalf("TestMulBackward failed ::: gradient of a in 'a * b' expected to be 5, not %g", a.grad)
	}
	if b.grad != a.data {
		t.Fatalf("TestMulBackward failed ::: gradient of b in 'a * b' expected to be 4, not %g", b.grad)
	}
}

func TestPowBackward(t *testing.T) {
	a := NewValue(2)
	exponent := NewValue(3.0)
	result := a.Pow(exponent)
	result.Backward()
	expectedGrad := 3 * math.Pow(2, 2) // 3 * a^(3-1)
	if a.grad != expectedGrad {
		t.Fatalf("TestPowBackward failed ::: gradient of a in 'a ^ 3' expected to be %g, not %g", expectedGrad, a.grad)
	}
}

func TestTanhBackward(t *testing.T) {
	a := NewValue(0.5)
	result := a.Tanh()
	result.Backward()
	expectedGrad := 1 - math.Pow(math.Tanh(0.5), 2) // 1 - tanh(a)^2
	if a.grad != expectedGrad {
		t.Fatalf("TestTanhBackward failed ::: gradient of a in 'tanh(a)' expected to be %g, not %g", expectedGrad, a.grad)
	}
}

func TestReluBackward(t *testing.T) {
	a := NewValue(-1)
	result := a.Relu()
	result.Backward()
	expectedGrad := 0.0 // gradient is 0 when a <= 0
	if a.grad != expectedGrad {
		t.Fatalf("TestReluBackward failed ::: gradient of a in 'relu(a)' expected to be %g, not %g", expectedGrad, a.grad)
	}
	b := NewValue(1)
	result = b.Relu()
	result.Backward()
	expectedGrad = 1.0 // gradient is 1 when a > 0
	if b.grad != expectedGrad {
		t.Fatalf("TestReluBackward failed ::: gradient of b in 'relu(b)' expected to be %g, not %g", expectedGrad, b.grad)
	}
}
