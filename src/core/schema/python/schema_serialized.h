// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Shared scaffolding for binding FlatBuffer tables to Python as encoded views.
//
// Python sees one class per table, backed by core::Serialized<Table>: reads go through
// the generated accessors straight into the buffer, and no object-API (`-T`) type is
// ever exposed. Structs (Pose, HandJoints, ...) are unaffected -- flatc emits one
// struct type for both APIs, so their existing bindings and the zero-copy NumPy views
// in schema_array_views.h keep working, reached through a table view that owns the
// buffer they alias.
//
// Construction from Python goes the other way: a constructor builds a `-T` as a local,
// encodes it, and returns the view. That keeps the encoder honest (it is the generated
// Pack, so the layout always matches the C++ readers) while the `-T` stays invisible.
//
// Every Tracked and Record wrapper has the same shape, so bind_tracked() and
// bind_record() below cover them; each schema's binding header only has to describe its
// own payload table.

#pragma once

#include <pybind11/pybind11.h>
#include <schema/serialized.hpp>
#include <schema/timestamp_generated.h>
#include <schema/tracked.hpp>

#include <memory>
#include <string>
#include <utility>

namespace py = pybind11;

namespace core
{

//! Python class for a table view. Chain the table's own fields onto the result.
template <typename T>
py::class_<Serialized<T>> serialized_class(py::module& m, const char* name, const char* doc)
{
    return py::class_<Serialized<T>>(m, name, doc)
        .def(
            "__bool__", [](const Serialized<T>& self) { return static_cast<bool>(self); },
            "False when the payload is absent.");
}

/*!
 * @brief Re-encodes a view's contents so it can be nested inside another table.
 *
 * FlatBuffers cannot splice one finished buffer into another, so composing a wrapper
 * around a payload the caller already built goes through the object API. This is the
 * one place that happens, and it is a construction-time cost only -- the read path
 * never unpacks.
 */
template <typename DataT>
std::shared_ptr<typename DataT::NativeTableType> to_native(const Serialized<DataT>& data)
{
    if (!data)
    {
        return nullptr;
    }
    return std::shared_ptr<typename DataT::NativeTableType>(data->UnPack());
}

/*!
 * @brief Binds a `Tracked` wrapper: `X()` for absent, `X(data)` to wrap a payload.
 *
 * `.data` yields None rather than an empty view when the payload is absent, so the
 * `if tracked.data is None` idiom these wrappers were built around keeps working.
 */
template <typename TrackedT, typename DataT>
void bind_tracked(py::module& m, const char* name, const char* data_name)
{
    const std::string data_doc = std::string("The ") + data_name + " payload, or None when absent.";
    serialized_class<TrackedT>(m, name, "Encoded tracker snapshot; .data is None when the payload is absent.")
        .def(py::init(
                 [](const Serialized<DataT>& data)
                 {
                     typename TrackedT::NativeTableType native;
                     native.data = to_native(data);
                     return pack<TrackedT>(native);
                 }),
             py::arg("data") = Serialized<DataT>(),
             "Encode a snapshot wrapping the given payload. Omit `data` for an absent snapshot.")
        .def_property_readonly(
            "data",
            [](const Serialized<TrackedT>& self) -> py::object
            {
                const DataT* data = payload(self);
                return data != nullptr ? py::cast(self.narrow(data)) : py::none();
            },
            data_doc.c_str())
        .def("__repr__",
             [name, data_name](const Serialized<TrackedT>& self)
             {
                 return std::string(name) +
                        "(data=" + (payload(self) != nullptr ? std::string(data_name) + "(...)" : "None") + ")";
             });
}

//! Binds a `Record` wrapper (an MCAP payload: data plus its capture timestamp).
template <typename RecordT, typename DataT>
void bind_record(py::module& m, const char* name, const char* data_name)
{
    serialized_class<RecordT>(m, name, "Encoded MCAP record: a payload plus the timestamp it was captured at.")
        .def(py::init<>(), "Construct an empty record (.data and .timestamp are None).")
        .def(py::init(
                 [](const Serialized<DataT>& data, const DeviceDataTimestamp& timestamp)
                 {
                     typename RecordT::NativeTableType native;
                     native.data = to_native(data);
                     native.timestamp = std::make_shared<DeviceDataTimestamp>(timestamp);
                     return pack<RecordT>(native);
                 }),
             py::arg("data"), py::arg("timestamp"), "Encode a record from a payload and its timestamp.")

        // Unlike Tracked, a Record cannot fold its no-arg form into defaults: `timestamp` is a
        // struct with no meaningful default, and an empty record is a distinct thing from one
        // stamped at time zero.
        .def_property_readonly(
            "data",
            [](const Serialized<RecordT>& self) -> py::object
            {
                const DataT* data = payload(self);
                return data != nullptr ? py::cast(self.narrow(data)) : py::none();
            },
            "The recorded payload, or None when absent.")
        .def_property_readonly(
            "timestamp", [](const Serialized<RecordT>& self) { return self ? self->timestamp() : nullptr; },
            py::return_value_policy::reference_internal, "Capture timestamp, or None when absent.")
        .def("__repr__",
             [name, data_name](const Serialized<RecordT>& self)
             {
                 return std::string(name) +
                        "(data=" + (payload(self) != nullptr ? std::string(data_name) + "(...)" : "None") + ")";
             });
}

} // namespace core
