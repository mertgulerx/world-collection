# Copyright 2026 Mert Güler
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Seeded random generators (T0 §3): numpy PCG64 through SeedSequence, so draws are reproducible."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def generator(seed: int, key: Sequence[int] = ()) -> np.random.Generator:
    """`numpy.random.Generator(PCG64(SeedSequence([seed, *key])))`."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, *key])))
