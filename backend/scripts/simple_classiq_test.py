# backend/scripts/simple_classiq_test.py

from classiq import (
    QArray,
    qfunc,
    create_model,
    synthesize,
    show,
    execute,
    Output,
    allocate,
    CX,
    H,
)


@qfunc
def main():
    """The main quantum function for the test.
    This circuit creates a Bell state.
    """
    x = QArray("x", size=1)
    y = QArray("y", size=1)

    allocate(1, x)
    allocate(1, y)

    H(x)
    CX(x, y)

    Output(x)
    Output(y)


def run_classiq_test():
    """Runs the basic functionality test for Classiq."""
    print("Testing basic Classiq functionality...")
    print("=" * 60)
    try:
        # Test model creation with the 'main' entry point
        model = create_model(main)
        print("✅ Model creation successful")

        # Test synthesis
        qprog = synthesize(model)
        print("✅ Synthesis successful")

    except Exception as e:
        import traceback
        print(f"❌ Error in basic test: {e}")
        print("Traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    run_classiq_test()