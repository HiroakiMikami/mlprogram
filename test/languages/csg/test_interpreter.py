import numpy as np
from mlprogram.interpreters import BatchedState
from mlprogram.languages.csg import show, Shape, Interpreter
from mlprogram.languages.csg \
    import Circle, Rectangle, Translation, Rotation, Union, Difference, \
    Reference


class TestShow(object):
    def test_str(self):
        c = np.zeros((3, 3), dtype=np.bool)
        assert "   \n   \n   \n" == show(c)
        c = np.eye(3, dtype=np.bool)
        assert "#  \n # \n  #\n" == show(c)


class TestShape(object):
    def test_render(self):
        shape = Shape(lambda x, y: x * y == 0)
        assert " # \n###\n # \n" == show(shape.render(3, 3))
        assert "  \n  \n" == show(shape.render(2, 2))

    def test_render_with_resolution(self):
        shape = Shape(lambda x, y: abs(x * y) < 0.5)
        assert " # \n###\n # \n" == show(shape.render(3, 3, 1))
        assert "  ##  \n  ##  \n######\n######\n  ##  \n  ##  \n" == \
            show(shape.render(3, 3, 2))
        assert "      \n      \n  ##  \n  ##  \n      \n      \n" == \
            show(shape.render(6, 6, 1))


class TestInterpreter(object):
    def test_circle(self):
        interpreter = Interpreter(1, 1, 1, False)
        assert "#\n" == show(interpreter.eval(Circle(1), [None])[0])

    def test_rectangle(self):
        code = Rectangle(1, 3)
        interpreter = Interpreter(3, 3, 1, False)
        assert " # \n # \n # \n" == show(interpreter.eval(code, [None])[0])

    def test_translation(self):
        code = Translation(2, 1, Rectangle(1, 3))
        interpreter = Interpreter(5, 5, 1, False)
        assert "    #\n    #\n    #\n     \n     \n" == \
            show(interpreter.eval(code, [None])[0])

    def test_rotation(self):
        code = Rotation(45, Rectangle(4, 1))
        interpreter = Interpreter(3, 3, 1, False)
        assert "  #\n # \n#  \n" == show(interpreter.eval(code, [None])[0])

    def test_union(self):
        code = Union(Rectangle(3, 1), Rectangle(1, 3))
        interpreter = Interpreter(3, 3, 1, False)
        assert " # \n###\n # \n" == show(interpreter.eval(code, [None])[0])

    def test_difference(self):
        code = Difference(Rectangle(1, 1), Rectangle(3, 1))
        interpreter = Interpreter(3, 3, 1, False)
        assert "   \n# #\n   \n" == show(interpreter.eval(code, [None])[0])

    def test_multiple_inputs(self):
        interpreter = Interpreter(1, 1, 1, False)
        results = interpreter.eval(Circle(1), [None, None])
        assert len(results) == 2

    def test_execute(self):
        ref0 = Rectangle(1, 1)
        ref1 = Rectangle(3, 1)
        ref2 = Difference(Reference(ref0), Reference(ref1))
        ref3 = Union(Rectangle(1, 1), Reference(ref2))
        state = BatchedState({}, {}, [])
        interpreter = Interpreter(3, 3, 1, False)

        state = interpreter.execute(ref0, [None], state)
        assert state.history == [ref0]
        assert set(state.environment.keys()) == set([ref0])
        assert state.type_environment[ref0] == "Rectangle"
        assert show(state.environment[ref0][0]) == "   \n # \n   \n"

        state = interpreter.execute(ref1, [None], state)
        assert state.history == [ref0, ref1]
        assert set(state.environment.keys()) == set([ref0, ref1])
        assert show(state.environment[ref1][0]) == "   \n###\n   \n"

        state = interpreter.execute(ref2, [None], state)
        assert state.history == [ref0, ref1, ref2]
        assert set(state.environment.keys()) == set([ref0, ref1, ref2])
        assert show(state.environment[ref2][0]) == "   \n# #\n   \n"

        state = interpreter.execute(ref3, [None], state)
        assert state.history == [ref0, ref1, ref2, ref3]
        assert set(state.environment.keys()) == set([ref0, ref1, ref2, ref3])
        assert show(state.environment[ref3][0]) == "   \n###\n   \n"

    def test_delete_used_variable(self):
        ref0 = Rectangle(1, 1)
        ref1 = Rectangle(3, 1)
        ref2 = Difference(Reference(ref0), Reference(ref1))
        ref3 = Union(Rectangle(1, 1), Reference(ref2))
        state = BatchedState({}, {}, [])
        interpreter = Interpreter(3, 3, 1, True)

        state = interpreter.execute(ref0, [None], state)
        assert set(state.environment.keys()) == set([ref0])

        state = interpreter.execute(ref1, [None], state)
        assert set(state.environment.keys()) == set([ref0, ref1])

        state = interpreter.execute(ref2, [None], state)
        assert set(state.environment.keys()) == set([ref2])

        state = interpreter.execute(ref3, [None], state)
        assert set(state.environment.keys()) == set([ref3])

    def test_execute_with_multiple_inputs(self):
        ref0 = Rectangle(1, 1)
        state = BatchedState({}, {}, [])
        interpreter = Interpreter(3, 3, 1, False)

        state = interpreter.execute(ref0, [None, None], state)
        assert len(state.environment[ref0]) == 2
