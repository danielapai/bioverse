"""
Example: Pre_Generator for steps that should run only once (e.g. reading a large
stellar catalog), then Generator for the rest of the pipeline.

:class:`~bioverse.generator.Pre_Generator` is a thin subclass of
:class:`~bioverse.generator.Generator` with the same methods
(:meth:`~bioverse.generator.Generator.insert_step`,
:meth:`~bioverse.generator.Generator.set_args`, etc.). Use it for clarity when
the first chunk of the program is intentionally run outside a loop or Monte Carlo
iteration.
"""
from bioverse.generator import Generator, Pre_Generator


def main():
    # Run the stellar catalog step once.
    pre = Pre_Generator()
    pre.insert_step("read_stellar_catalog")
    pre.set_args(d_max=30.0)

    stars = pre.generate()

    # Continue the default imaging pipeline from step 1 onward, reusing `stars`
    # as the initial table (skip the duplicate catalog read).
    gen = Generator("imaging")
    sample = gen.generate(d=stars, idx_start=1)

    return sample


if __name__ == "__main__":
    main()
