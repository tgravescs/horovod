# Copyright 2017 onwards, fast.ai, Inc.
# Modifications copyright (C) 2019 Uber Technologies, Inc.
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
# ==============================================================================

import argparse
import datetime
import os
from distutils.version import LooseVersion

if __name__ == '__main__':
    # ================ #
    # DATA PREPARATION #
    # ================ #

    # Location of discovery script on local filesystem.
    DISCOVERY_SCRIPT = 'get_gpu_resources.sh'

    print('================')
    print('Data preparation')
    print('================')

    print("Tom before")
    import pyarrow as pa
    #os.environ['CLASSPATH'] = "/home/tgraves/hadoop2confs:/home/tgraves/hadoop-3.1.3/share/hadoop/common/lib/*:/home/tgraves/hadoop-3.1.3/share/hadoop/common/*:/home/tgraves/hadoop-3.1.3/share/hadoop/hdfs:/home/tgraves/hadoop-3.1.3/share/hadoop/hdfs/lib/*:/home/tgraves/hadoop-3.1.3/share/hadoop/hdfs/*"
    hdfs_kwargs = dict(host="spark-egx-10",
                      port=9000,
                      user="tgraves",
                      kerb_ticket=None,
                      extra_conf=None)
    fs = pa.hdfs.connect(**hdfs_kwargs)
    res = fs.exists("/user/tgraves")
    #store = Store.create("hdfs://10.136.6.6:8020/user/tgraves")
    #res = store.exists("/user/tgraves")
    print("Tom before %s" % res)

