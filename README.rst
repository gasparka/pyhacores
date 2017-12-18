pyhacores
=========


.. image:: https://img.shields.io/pypi/v/pyhacores.svg
        :target: https://pypi.python.org/pypi/pyhacores

.. image:: https://travis-ci.org/gasparka/pyhacores.svg?branch=develop
    :target: https://travis-ci.org/gasparka/pyhacores

.. image:: https://pyup.io/repos/github/gasparka/pyhacores/shield.svg
     :target: https://pyup.io/repos/github/gasparka/pyhacores/
     :alt: Updates

.. image:: https://coveralls.io/repos/github/gasparka/pyhacores/badge.svg?branch=develop
    :target: https://coveralls.io/github/gasparka/pyhacores?branch=develop



Cores written with `Pyha <https://github.com/gasparka/pyha/>`_

* Free software: Apache Software License 2.0

Available cores:

- `CORDIC`_ : CORDIC core.
- `NCO`_: Numerically controlled oscillator, based on CORDIC.
- `ToPolar`_: Converts IQ to polar form, returning equal to ``np.abs()`` and ``np.angle()/pi``.
- `Angle`_: Equal to ``np.angle()/pi``.
- `Abs`_: Equal to ``np.abs()``.
- `FIR`_: Transposed form FIR filter, equal to ``scipy.signal.lfilter()``.
- `MovingAverage`_: Hardware friendly implementation of moving average filter.
- `DCRemoval`_: Filter out DC component, internally uses chained `MovingAverage`_ blocks.
- `CRC16`_: Calculate 16 bit CRC, galois based.
- `HeaderCorrelator`_: Correlate against 16 bit package header.
- `QuadratureDemodulator`_: Demodulate FM, FSK, GMSK...
- `BladeRFSource`_: Convert BladeRF style I/Q (4 downto -11) into Pyha Complex (0 downto -17) type
- `BladeRFSink`_: Convert Pyha Complex style (0 downto -17) into BladeRF style I/Q (4 downto -11)
- `ComplexConjugate`_: Complex conjugate with output register
- `ComplexMultiply`_: Complex multiplycate with output register


.. _CORDIC: https://github.com/gasparka/pyhacores/blob/develop/pyhacores/cordic/cordic_core.py
.. _NCO: https://github.com/gasparka/pyhacores/blob/develop/pyhacores/cordic/nco.py
.. _ToPolar: https://github.com/gasparka/pyhacores/blob/develop/pyhacores/cordic/to_polar.py
.. _Abs: https://github.com/gasparka/pyhacores/blob/develop/pyhacores/cordic/to_polar.py
.. _Angle: https://github.com/gasparka/pyhacores/blob/develop/pyhacores/cordic/to_polar.py
.. _FIR: https://github.com/gasparka/pyhacores/blob/develop/pyhacores/filter/fir.py
.. _MovingAverage: https://github.com/gasparka/pyhacores/blob/develop/pyhacores/filter/moving_average.py
.. _DCRemoval: https://github.com/gasparka/pyhacores/blob/develop/pyhacores/filter/dc_removal.py
.. _CRC16: https://github.com/gasparka/pyhacores/blob/develop/pyhacores/packet/crc16.py
.. _HeaderCorrelator: https://github.com/gasparka/pyhacores/blob/develop/pyhacores/packet/header_correlator.py
.. _QuadratureDemodulator: https://github.com/gasparka/pyhacores/blob/develop/pyhacores/radio/quadrature_demodulator.py
.. _BladeRFSource: https://github.com/gasparka/pyhacores/blob/develop/pyhacores/util/blade_rf.py
.. _BladeRFSink: https://github.com/gasparka/pyhacores/blob/develop/pyhacores/util/blade_rf.py
.. _ComplexConjugate: https://github.com/gasparka/pyhacores/blob/develop/pyhacores/util/complex.py
.. _ComplexMultiply: https://github.com/gasparka/pyhacores/blob/develop/pyhacores/util/complex.py

