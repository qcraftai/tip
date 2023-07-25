# Modifications Copyright (C) 2023 QCraft, Inc., Wei-Xin Li, Xiaodong Yang.
#
# ------------------------------------------------------------------------
#
# Copyright 2020 NVIDIA CORPORATION, Jonah Philion, Amlan Kar, Sanja Fidler
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

from fire import Fire
from trans_idealism_planner import explore

if __name__ == "__main__":
    Fire(
        {
            "false_neg_viz": explore.false_neg_viz,
            "false_pos_viz": explore.false_pos_viz,
            "eval_test": explore.eval_test,
            "score_distribution_plot": explore.score_distribution_plot,
        }
    )
