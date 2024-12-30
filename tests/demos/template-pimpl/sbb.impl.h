#ifndef SBB_IMPL_H__
#define SBB_IMPL_H__

#include <fmt/format.h>

#include <iostream>
#include <typeinfo>

#include "sbb.h"

template <int dim, typename T>
class SBB<dim, T>::impl
{
  SBB<dim, T> *self_;
  T            data_;

public:
  impl(SBB<dim, T> *self)
    : self_(self)
    , data_()
  {
    // fmt::println("SBB::impl::impl(): self={}", (void *)self_);
  }
  ~impl()
  {
    // fmt::println("SBB::impl::~impl(): self={}", (void *)self_);
  }

  impl(const impl &other) = delete;
  impl &
  operator=(const impl &other) = delete;

  void
  chown(SBB<dim, T> *self)
  {
    self_ = self;
  }
  impl &
  assign(const impl &other)
  {
    data_ = other.data_;
    return *this;
  }

  T
  get_data() const
  {
    return data_;
  }
  void
  set_data(T data)
  {
    data_ = data;
  }
};

template <int dim, typename T>
T SBB<dim, T>::svar;

template <int dim, typename T>
SBB<dim, T>::SBB()
  : impl_{std::make_unique<impl>(this)}
{
  // fmt::println("SBB::SBB(): this={}", (void *)this);
}

template <int dim, typename T>
SBB<dim, T>::~SBB()
{
  // fmt::println("SBB::~SBB(): this={}", (void *)this);
}

template <int dim, typename T>
SBB<dim, T>::SBB(const SBB<dim, T> &other)
  : impl_{std::make_unique<impl>(this)}
{
  // fmt::println("SBB::SBB(const SBB &other): this={}, other={}",
  //              (void *)this,
  //              (void *)&other);
  impl_->assign(*other.impl_);
}

template <int dim, typename T>
SBB<dim, T> &
SBB<dim, T>::operator=(const SBB<dim, T> &other)
{
  // fmt::println("SBB::operator=(const SBB &other): this={}, other={}",
  //              (void *)this,
  //              (void *)&other);
  if (this != &other)
    {
      impl_->assign(*other.impl_);
    }
  return *this;
}

template <int dim, typename T>
SBB<dim, T>::SBB(SBB<dim, T> &&other)
  : impl_{std::move(other.impl_)}
{
  // fmt::println("SBB::SBB(SBB &&other): this={}, other={}",
  //              (void *)this,
  //              (void *)&other);
  impl_->chown(this);
}

template <int dim, typename T>
SBB<dim, T> &
SBB<dim, T>::operator=(SBB<dim, T> &&other)
{
  // fmt::println("SBB::operator=(SBB &&other): this={}, other={}",
  //              (void *)this,
  //              (void *)&other);
  if (this != &other)
    {
      impl_ = std::move(other.impl_);
      impl_->chown(this);
    }
  return *this;
}

template <int dim, typename T>
SBB<dim, T>::SBB(const T &&value)
  : impl_{std::make_unique<impl>(this)}
{
  // fmt::println("SBB::SBB(const T &&value): this={}", (void *)this);
  impl_->set_data(value);
}

template <int dim, typename T>
template <int dim1>
SBB<dim, T>::SBB(const Mapping<dim1, dim> &)
  : impl_{std::make_unique<impl>(this)}
{
  impl_->set_data(dim1);
}

template <int dim, typename T>
T
SBB<dim, T>::get_value() const
{
  return impl_->get_data();
}

template <int dim, typename T>
void
SBB<dim, T>::set_value(T value)
{
  impl_->set_data(value);
}

template <int dim, typename T>
void
SBB<dim, T>::print() const
{
  std::cout << *this << std::endl;
}

template <int dim, typename T>
size_t
SBB<dim, T>::get_size() const
{
  return sizeof(impl_->get_data());
}

template <int dim, typename T>
std::string
SBB<dim, T>::to_string() const
{
  return fmt::format("SBB(dim={}, type={}): {}",
                     dim,
                     typeid(T).name(),
                     impl_->get_data());
}

#endif // SBB_IMPL_H__
