// Author: Paulo R. A. Gilz
//
// Computed-parameter expression compiler.
//
// Users define derived channels such as
//     LOAD_FACTOR = NZ * 9.80665
//     DRIFT       = HEADING - HEADING[TIME=0.0]
//     ENVELOPE    = (MACH > 0.8) * ALPHA
// and plot them like any other parameter.
//
// Flat bytecode rather than a virtual-dispatch AST, for three reasons:
//
//  * Compilation happens once per formula, not once per simulation. Binding raw
//    `const float*` pointers into AST nodes forces a re-parse for every
//    simulation -- and those pointers dangle the moment a std::vector
//    reallocates during a later column load. Here the bytecode stores variable
//    *slots*, and data is bound at evaluation.
//  * No per-node allocation, no virtual call, and the instruction stream walks
//    linearly through cache.
//  * Scalar tracking. `2 * 9.81` collapses to a single float instead of filling
//    a million-element buffer, and a scalar operand of a vector operation is
//    applied without materialising it.
//  * Adds functions (abs, sqrt, trig, exp, log, min, max, atan2, clamp...),
//    unary minus, modulo, `!=`, `<=`, `>=` and logical not -- all missing before.
//  * Division by zero yields NaN rather than inf, so bad samples break the line
//    instead of collapsing the plot's y-range to +/- infinity.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace aoe {

// Supplies sample data for a variable name. Returns nullptr when unknown; sets
// count to the number of samples otherwise.
using ChannelResolver =
    std::function<const float*(const std::string& name, std::size_t& count)>;

class Formula {
public:
    Formula() = default;

    // Parses `expression`. On failure returns an invalid Formula and fills
    // `error` with a human-readable message including the offending position.
    static Formula compile(const std::string& expression, std::string* error);

    bool valid() const noexcept { return valid_; }
    const std::string& source() const noexcept { return source_; }

    // Every variable name referenced, in first-use order. The caller uses this
    // to preload exactly the columns the formula needs, in one batch.
    const std::vector<std::string>& variables() const noexcept { return variables_; }

    // True when the expression references no channels at all, so its value is
    // the same for every simulation.
    bool is_constant() const noexcept { return variables_.empty(); }

    // Evaluates into `out`. `count` is the sample count to produce; pass 0 to
    // use the shortest resolved channel, which is what plotting wants (mixing
    // channels of different lengths otherwise reads past the shorter one).
    // Returns false and sets `error` if a variable cannot be resolved.
    bool evaluate(const ChannelResolver& resolver, std::size_t count,
                  std::vector<float>& out, std::string* error) const;

    // Checks that the expression parses and that every variable it names is
    // accepted by `known`. Used for live validation in the formula editor
    // without touching any data.
    static bool validate(const std::string& expression,
                         const std::function<bool(const std::string&)>& known,
                         std::string* error);

private:
    enum class Op : std::uint8_t {
        Const, Var, Lookup,
        Add, Sub, Mul, Div, Mod, Pow,
        Neg, Not,
        Lt, Gt, Le, Ge, Eq, Ne, And, Or,
        Fn1, Fn2, Fn3,
        // Whole-series operations (derivative, filters...). `a` carries the
        // argument count, which varies per function unlike Fn1/Fn2/Fn3.
        Series
    };

    struct Instr {
        Op op = Op::Const;
        std::uint8_t fn = 0;
        std::int32_t a = -1;  // variable slot / function id
        std::int32_t b = -1;  // condition variable slot for Lookup
        float value = 0.0f;   // literal, or the condition value for Lookup
    };

    friend class Parser;

    // Scalar kernel for every binary opcode. Defined out of line so the
    // evaluator's inner loops stay small enough to inline.
    static float binary_apply(Op op, float a, float b);

    std::vector<Instr> code_;
    std::vector<std::string> variables_;
    std::string source_;
    bool valid_ = false;
};

}  // namespace aoe
