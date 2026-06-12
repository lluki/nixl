/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef OBJ_PLUGIN_AWS_SDK_INIT_H
#define OBJ_PLUGIN_AWS_SDK_INIT_H

#include <aws/core/Aws.h>
#include <aws/core/utils/logging/ConsoleLogSystem.h>
#include <aws/core/utils/logging/DefaultCRTLogSystem.h>
#include <aws/core/utils/logging/LogLevel.h>
#include <memory>
#include <mutex>
#include <cstdlib>
#include <strings.h>

namespace nixl_s3_utils {

// Local NIXL addition: map the NIXL_OBJ_AWS_LOG env var to an AWS SDK log level.
// Off (default) preserves upstream behaviour (no SDK logging). Debug/Trace make
// the AWS SDK log every HTTP request (and Trace, the wire-level detail) to stderr.
inline Aws::Utils::Logging::LogLevel
nixlParseAwsLogLevel(const char *s) {
    using L = Aws::Utils::Logging::LogLevel;
    if (!s || !*s) return L::Off;
    if (!strcasecmp(s, "Trace")) return L::Trace;
    if (!strcasecmp(s, "Debug")) return L::Debug;
    if (!strcasecmp(s, "Info")) return L::Info;
    if (!strcasecmp(s, "Warn")) return L::Warn;
    if (!strcasecmp(s, "Error")) return L::Error;
    if (!strcasecmp(s, "Fatal")) return L::Fatal;
    return L::Off;
}

/**
 * Initialize the AWS SDK in a thread-safe manner.
 * This function uses std::call_once to ensure that Aws::InitAPI is called
 * exactly once, even in multi-threaded environments or when multiple S3 clients
 * are created.
 *
 * The AWS SDK is automatically shut down at program exit via std::atexit.
 */
inline void
initAWSSDK() {
    static std::once_flag aws_init_flag;
    static Aws::SDKOptions *aws_options = nullptr;

    std::call_once(aws_init_flag, []() {
        aws_options = new Aws::SDKOptions();

        // NIXL addition: enable AWS SDK logging when NIXL_OBJ_AWS_LOG is set.
        // Routes to stderr (ConsoleLogSystem) for both the standard S3 client
        // and the CRT client (crt_logger), so every HTTP request is visible.
        const Aws::Utils::Logging::LogLevel level =
            nixlParseAwsLogLevel(std::getenv("NIXL_OBJ_AWS_LOG"));
        if (level != Aws::Utils::Logging::LogLevel::Off) {
            aws_options->loggingOptions.logLevel = level;
            aws_options->loggingOptions.logger_create_fn = [level]() {
                return std::make_shared<Aws::Utils::Logging::ConsoleLogSystem>(level);
            };
            aws_options->loggingOptions.crt_logger_create_fn = [level]() {
                return std::make_shared<Aws::Utils::Logging::DefaultCRTLogSystem>(level);
            };
        }

        Aws::InitAPI(*aws_options);

        // Register cleanup at program exit
        std::atexit([]() {
            Aws::ShutdownAPI(*aws_options);
            delete aws_options;
        });
    });
}

} // namespace nixl_s3_utils

#endif // OBJ_PLUGIN_AWS_SDK_INIT_H
