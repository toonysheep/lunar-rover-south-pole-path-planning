# Third-party provenance

`vendor/gym/` is the historical Gym 0.18.0 package retained from the research workspace. Its MIT license and original notices are preserved in `vendor/gym/LICENSE.md`; some asset-related notices are retained alongside their source directories. The unused nested duplicate Gym development checkout is excluded.

`vendor/gym_minigrid/` is a research-modified legacy MiniGrid codebase derived from the Gym MiniGrid project (`https://github.com/maximecb/gym-minigrid`). The upstream project is distributed under the Apache License 2.0; the standard license text is restored at `vendor/gym_minigrid/LICENSE`. Modifications include lunar environmental input, stay/eight-direction actions, observations, rewards and thermal/battery calculations. The original workspace did not preserve a reliable upstream commit identifier, so the exact base revision remains to be documented before the permanent public release.

Names in upstream notices identify the original maintainers of those third-party components. A repository-level license for the original research contributions and data should be added before the permanent public release.
