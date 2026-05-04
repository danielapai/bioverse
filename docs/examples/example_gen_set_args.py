"""
Example: set generator keyword arguments with set_args before running generate().

Calling set_args() before generate() records the intended settings on each step
so you can inspect the Generator later (e.g. print(generator) or save the
Generator) and see what values were used.

See the Sphinx page *bioverse.functions module* (API reference) for all
built-in step functions.
"""
from bioverse.generator import Generator


def main():
    generator = Generator("imaging")

    # Pass a dict of keyword arguments shared by one or more steps (unpack with **).
    # This is clearer than repeating the same kwargs on every generate() call.
    gen_kwargs = {
        "d_max": 30.0,
        "f_eta": 1.0,
    }
    generator.set_args(**gen_kwargs)

    # Optional: start from an existing Table (e.g. output of a Pre_Generator).
    # Omit `d` to begin from an empty table.
    sample = generator.generate()

    # The generator object still reflects the arguments you set.
    print(generator)
    return sample


if __name__ == "__main__":
    main()
