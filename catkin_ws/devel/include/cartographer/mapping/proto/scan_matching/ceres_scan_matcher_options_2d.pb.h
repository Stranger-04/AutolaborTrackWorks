// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: cartographer/mapping/proto/scan_matching/ceres_scan_matcher_options_2d.proto

#ifndef PROTOBUF_cartographer_2fmapping_2fproto_2fscan_5fmatching_2fceres_5fscan_5fmatcher_5foptions_5f2d_2eproto__INCLUDED
#define PROTOBUF_cartographer_2fmapping_2fproto_2fscan_5fmatching_2fceres_5fscan_5fmatcher_5foptions_5f2d_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3005000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3005000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
#include "cartographer/common/proto/ceres_solver_options.pb.h"
// @@protoc_insertion_point(includes)

namespace protobuf_cartographer_2fmapping_2fproto_2fscan_5fmatching_2fceres_5fscan_5fmatcher_5foptions_5f2d_2eproto {
// Internal implementation detail -- do not use these members.
struct TableStruct {
  static const ::google::protobuf::internal::ParseTableField entries[];
  static const ::google::protobuf::internal::AuxillaryParseTableField aux[];
  static const ::google::protobuf::internal::ParseTable schema[1];
  static const ::google::protobuf::internal::FieldMetadata field_metadata[];
  static const ::google::protobuf::internal::SerializationTable serialization_table[];
  static const ::google::protobuf::uint32 offsets[];
};
void AddDescriptors();
void InitDefaultsCeresScanMatcherOptions2DImpl();
void InitDefaultsCeresScanMatcherOptions2D();
inline void InitDefaults() {
  InitDefaultsCeresScanMatcherOptions2D();
}
}  // namespace protobuf_cartographer_2fmapping_2fproto_2fscan_5fmatching_2fceres_5fscan_5fmatcher_5foptions_5f2d_2eproto
namespace cartographer {
namespace mapping {
namespace scan_matching {
namespace proto {
class CeresScanMatcherOptions2D;
class CeresScanMatcherOptions2DDefaultTypeInternal;
extern CeresScanMatcherOptions2DDefaultTypeInternal _CeresScanMatcherOptions2D_default_instance_;
}  // namespace proto
}  // namespace scan_matching
}  // namespace mapping
}  // namespace cartographer
namespace cartographer {
namespace mapping {
namespace scan_matching {
namespace proto {

// ===================================================================

class CeresScanMatcherOptions2D : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:cartographer.mapping.scan_matching.proto.CeresScanMatcherOptions2D) */ {
 public:
  CeresScanMatcherOptions2D();
  virtual ~CeresScanMatcherOptions2D();

  CeresScanMatcherOptions2D(const CeresScanMatcherOptions2D& from);

  inline CeresScanMatcherOptions2D& operator=(const CeresScanMatcherOptions2D& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  CeresScanMatcherOptions2D(CeresScanMatcherOptions2D&& from) noexcept
    : CeresScanMatcherOptions2D() {
    *this = ::std::move(from);
  }

  inline CeresScanMatcherOptions2D& operator=(CeresScanMatcherOptions2D&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  static const ::google::protobuf::Descriptor* descriptor();
  static const CeresScanMatcherOptions2D& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const CeresScanMatcherOptions2D* internal_default_instance() {
    return reinterpret_cast<const CeresScanMatcherOptions2D*>(
               &_CeresScanMatcherOptions2D_default_instance_);
  }
  static PROTOBUF_CONSTEXPR int const kIndexInFileMessages =
    0;

  void Swap(CeresScanMatcherOptions2D* other);
  friend void swap(CeresScanMatcherOptions2D& a, CeresScanMatcherOptions2D& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline CeresScanMatcherOptions2D* New() const PROTOBUF_FINAL { return New(NULL); }

  CeresScanMatcherOptions2D* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CopyFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void MergeFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void CopyFrom(const CeresScanMatcherOptions2D& from);
  void MergeFrom(const CeresScanMatcherOptions2D& from);
  void Clear() PROTOBUF_FINAL;
  bool IsInitialized() const PROTOBUF_FINAL;

  size_t ByteSizeLong() const PROTOBUF_FINAL;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) PROTOBUF_FINAL;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const PROTOBUF_FINAL;
  int GetCachedSize() const PROTOBUF_FINAL { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const PROTOBUF_FINAL;
  void InternalSwap(CeresScanMatcherOptions2D* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const PROTOBUF_FINAL;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // .cartographer.common.proto.CeresSolverOptions ceres_solver_options = 9;
  bool has_ceres_solver_options() const;
  void clear_ceres_solver_options();
  static const int kCeresSolverOptionsFieldNumber = 9;
  const ::cartographer::common::proto::CeresSolverOptions& ceres_solver_options() const;
  ::cartographer::common::proto::CeresSolverOptions* release_ceres_solver_options();
  ::cartographer::common::proto::CeresSolverOptions* mutable_ceres_solver_options();
  void set_allocated_ceres_solver_options(::cartographer::common::proto::CeresSolverOptions* ceres_solver_options);

  // double occupied_space_weight = 1;
  void clear_occupied_space_weight();
  static const int kOccupiedSpaceWeightFieldNumber = 1;
  double occupied_space_weight() const;
  void set_occupied_space_weight(double value);

  // double translation_weight = 2;
  void clear_translation_weight();
  static const int kTranslationWeightFieldNumber = 2;
  double translation_weight() const;
  void set_translation_weight(double value);

  // double rotation_weight = 3;
  void clear_rotation_weight();
  static const int kRotationWeightFieldNumber = 3;
  double rotation_weight() const;
  void set_rotation_weight(double value);

  // @@protoc_insertion_point(class_scope:cartographer.mapping.scan_matching.proto.CeresScanMatcherOptions2D)
 private:

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::cartographer::common::proto::CeresSolverOptions* ceres_solver_options_;
  double occupied_space_weight_;
  double translation_weight_;
  double rotation_weight_;
  mutable int _cached_size_;
  friend struct ::protobuf_cartographer_2fmapping_2fproto_2fscan_5fmatching_2fceres_5fscan_5fmatcher_5foptions_5f2d_2eproto::TableStruct;
  friend void ::protobuf_cartographer_2fmapping_2fproto_2fscan_5fmatching_2fceres_5fscan_5fmatcher_5foptions_5f2d_2eproto::InitDefaultsCeresScanMatcherOptions2DImpl();
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// CeresScanMatcherOptions2D

// double occupied_space_weight = 1;
inline void CeresScanMatcherOptions2D::clear_occupied_space_weight() {
  occupied_space_weight_ = 0;
}
inline double CeresScanMatcherOptions2D::occupied_space_weight() const {
  // @@protoc_insertion_point(field_get:cartographer.mapping.scan_matching.proto.CeresScanMatcherOptions2D.occupied_space_weight)
  return occupied_space_weight_;
}
inline void CeresScanMatcherOptions2D::set_occupied_space_weight(double value) {
  
  occupied_space_weight_ = value;
  // @@protoc_insertion_point(field_set:cartographer.mapping.scan_matching.proto.CeresScanMatcherOptions2D.occupied_space_weight)
}

// double translation_weight = 2;
inline void CeresScanMatcherOptions2D::clear_translation_weight() {
  translation_weight_ = 0;
}
inline double CeresScanMatcherOptions2D::translation_weight() const {
  // @@protoc_insertion_point(field_get:cartographer.mapping.scan_matching.proto.CeresScanMatcherOptions2D.translation_weight)
  return translation_weight_;
}
inline void CeresScanMatcherOptions2D::set_translation_weight(double value) {
  
  translation_weight_ = value;
  // @@protoc_insertion_point(field_set:cartographer.mapping.scan_matching.proto.CeresScanMatcherOptions2D.translation_weight)
}

// double rotation_weight = 3;
inline void CeresScanMatcherOptions2D::clear_rotation_weight() {
  rotation_weight_ = 0;
}
inline double CeresScanMatcherOptions2D::rotation_weight() const {
  // @@protoc_insertion_point(field_get:cartographer.mapping.scan_matching.proto.CeresScanMatcherOptions2D.rotation_weight)
  return rotation_weight_;
}
inline void CeresScanMatcherOptions2D::set_rotation_weight(double value) {
  
  rotation_weight_ = value;
  // @@protoc_insertion_point(field_set:cartographer.mapping.scan_matching.proto.CeresScanMatcherOptions2D.rotation_weight)
}

// .cartographer.common.proto.CeresSolverOptions ceres_solver_options = 9;
inline bool CeresScanMatcherOptions2D::has_ceres_solver_options() const {
  return this != internal_default_instance() && ceres_solver_options_ != NULL;
}
inline const ::cartographer::common::proto::CeresSolverOptions& CeresScanMatcherOptions2D::ceres_solver_options() const {
  const ::cartographer::common::proto::CeresSolverOptions* p = ceres_solver_options_;
  // @@protoc_insertion_point(field_get:cartographer.mapping.scan_matching.proto.CeresScanMatcherOptions2D.ceres_solver_options)
  return p != NULL ? *p : *reinterpret_cast<const ::cartographer::common::proto::CeresSolverOptions*>(
      &::cartographer::common::proto::_CeresSolverOptions_default_instance_);
}
inline ::cartographer::common::proto::CeresSolverOptions* CeresScanMatcherOptions2D::release_ceres_solver_options() {
  // @@protoc_insertion_point(field_release:cartographer.mapping.scan_matching.proto.CeresScanMatcherOptions2D.ceres_solver_options)
  
  ::cartographer::common::proto::CeresSolverOptions* temp = ceres_solver_options_;
  ceres_solver_options_ = NULL;
  return temp;
}
inline ::cartographer::common::proto::CeresSolverOptions* CeresScanMatcherOptions2D::mutable_ceres_solver_options() {
  
  if (ceres_solver_options_ == NULL) {
    ceres_solver_options_ = new ::cartographer::common::proto::CeresSolverOptions;
  }
  // @@protoc_insertion_point(field_mutable:cartographer.mapping.scan_matching.proto.CeresScanMatcherOptions2D.ceres_solver_options)
  return ceres_solver_options_;
}
inline void CeresScanMatcherOptions2D::set_allocated_ceres_solver_options(::cartographer::common::proto::CeresSolverOptions* ceres_solver_options) {
  ::google::protobuf::Arena* message_arena = GetArenaNoVirtual();
  if (message_arena == NULL) {
    delete reinterpret_cast< ::google::protobuf::MessageLite*>(ceres_solver_options_);
  }
  if (ceres_solver_options) {
    ::google::protobuf::Arena* submessage_arena = NULL;
    if (message_arena != submessage_arena) {
      ceres_solver_options = ::google::protobuf::internal::GetOwnedMessage(
          message_arena, ceres_solver_options, submessage_arena);
    }
    
  } else {
    
  }
  ceres_solver_options_ = ceres_solver_options;
  // @@protoc_insertion_point(field_set_allocated:cartographer.mapping.scan_matching.proto.CeresScanMatcherOptions2D.ceres_solver_options)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace proto
}  // namespace scan_matching
}  // namespace mapping
}  // namespace cartographer

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_cartographer_2fmapping_2fproto_2fscan_5fmatching_2fceres_5fscan_5fmatcher_5foptions_5f2d_2eproto__INCLUDED
