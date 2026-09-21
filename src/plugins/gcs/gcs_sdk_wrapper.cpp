/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "gcs_sdk_wrapper.h"
#include <google/cloud/storage/client.h>
#include <iostream>
#include <memory>

namespace gcs = ::google::cloud::storage;

struct nixl_gcs_client {
    gcs::Client client;
};

extern "C" nixl_gcs_client *
nixl_gcs_client_create() {
    try {
        return new nixl_gcs_client{};
    }
    catch (const std::exception &e) {
        std::cerr << "GCS client initialization failed: " << e.what() << '\n';
        return nullptr;
    }
}

extern "C" void
nixl_gcs_client_destroy(nixl_gcs_client *client) {
    delete client;
}

extern "C" bool
nixl_gcs_put(nixl_gcs_client *client,
             const char *bucket,
             const char *object,
             const void *data,
             std::size_t length,
             std::uint64_t offset) {
    if (offset != 0) {
        return false;
    }
    try {
        auto stream = client->client.WriteObject(bucket, object);
        stream.write(static_cast<const char *>(data), length);
        stream.Close();
        const auto metadata = std::move(stream).metadata();
        if (!metadata) {
            std::cerr << "GCS write failed: " << metadata.status().message() << '\n';
            return false;
        }
        return true;
    }
    catch (const std::exception &e) {
        std::cerr << "GCS write exception: " << e.what() << '\n';
        return false;
    }
}

extern "C" bool
nixl_gcs_get(nixl_gcs_client *client,
             const char *bucket,
             const char *object,
             void *data,
             std::size_t length,
             std::uint64_t offset) {
    try {
        auto stream =
            client->client.ReadObject(bucket,
                                      object,
                                      gcs::ReadRange(static_cast<std::int64_t>(offset),
                                                     static_cast<std::int64_t>(offset + length)));
        stream.read(static_cast<char *>(data), length);
        const auto got = stream.gcount();
        stream.Close();
        if (got != static_cast<std::streamsize>(length) || !stream.status().ok()) {
            std::cerr << "GCS read failed/short: " << stream.status().message() << " bytes=" << got
                      << "/" << length << '\n';
            return false;
        }
        return true;
    }
    catch (const std::exception &e) {
        std::cerr << "GCS read exception: " << e.what() << '\n';
        return false;
    }
}

extern "C" int
nixl_gcs_exists(nixl_gcs_client *client, const char *bucket, const char *object) {
    const auto metadata = client->client.GetObjectMetadata(bucket, object);
    if (metadata) {
        return 1;
    }
    if (metadata.status().code() == google::cloud::StatusCode::kNotFound) {
        return 0;
    }
    std::cerr << "GCS metadata query failed: " << metadata.status().message() << '\n';
    return -1;
}
