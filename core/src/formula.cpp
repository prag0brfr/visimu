// Author: Paulo R. A. Gilz
#include "aoe/formula.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace aoe {
namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

// Lookup modes, carried in Instr::fn.
constexpr std::uint8_t kLookupNearest = 0;
constexpr std::uint8_t kLookupFirst = 1;
constexpr std::uint8_t kLookupLast = 2;

struct FnDef {
    const char* name;
    int arity;
    std::uint8_t id;
};

// Function ids are grouped by arity so the evaluator can switch on id alone.
enum : std::uint8_t {
    F_ABS = 0, F_SQRT, F_SIN, F_COS, F_TAN, F_ASIN, F_ACOS, F_ATAN,
    F_EXP, F_LOG, F_LOG10, F_FLOOR, F_CEIL, F_ROUND, F_SIGN, F_DEG, F_RAD,
    F_MIN = 0, F_MAX, F_ATAN2, F_POW, F_MOD, F_HYPOT,
    F_CLAMP = 0, F_IF
};

const FnDef kFn1[] = {
    {"abs", 1, F_ABS},     {"sqrt", 1, F_SQRT},   {"sin", 1, F_SIN},
    {"cos", 1, F_COS},     {"tan", 1, F_TAN},     {"asin", 1, F_ASIN},
    {"acos", 1, F_ACOS},   {"atan", 1, F_ATAN},   {"exp", 1, F_EXP},
    {"log", 1, F_LOG},     {"ln", 1, F_LOG},      {"log10", 1, F_LOG10},
    {"floor", 1, F_FLOOR}, {"ceil", 1, F_CEIL},   {"round", 1, F_ROUND},
    {"sign", 1, F_SIGN},   {"deg", 1, F_DEG},     {"rad", 1, F_RAD},
};

const FnDef kFn2[] = {
    {"min", 2, F_MIN},     {"max", 2, F_MAX},   {"atan2", 2, F_ATAN2},
    {"pow", 2, F_POW},     {"mod", 2, F_MOD},   {"hypot", 2, F_HYPOT},
};

const FnDef kFn3[] = {
    {"clamp", 3, F_CLAMP}, {"if", 3, F_IF},
};

// Whole-series functions. Unlike everything above these cannot be evaluated one
// sample at a time -- a derivative needs its neighbours, a filter needs its own
// previous output -- so they get their own opcode and operate on the finished
// buffer.
enum : std::uint8_t {
    S_DERIV = 0, S_INTEG, S_LOWPASS, S_HIGHPASS, S_MOVAVG, S_DELTA,
    S_MIN, S_MAX, S_MEAN,
};

const FnDef kFnSeries[] = {
    {"deriv", 2, S_DERIV},      // deriv(Y, X)         dY/dX
    {"integ", 2, S_INTEG},      // integ(Y, X)         cumulative trapezoid
    {"lowpass", 3, S_LOWPASS},  // lowpass(Y, X, fc)   first-order, fc in Hz
    {"highpass", 3, S_HIGHPASS},
    {"movavg", 2, S_MOVAVG},    // movavg(Y, N)        centred, N samples
    {"delta", 1, S_DELTA},      // delta(Y)            Y - Y[0]

    // Reductions: one argument, and the answer is a single value held flat
    // across the run, so NZ / max(NZ) normalises against the peak of the same
    // run and MACH - min(MACH) is the rise from the slowest point.
    //
    // These share their names with the two-argument pointwise min and max --
    // min(A, B) compares two channels sample by sample, min(A) collapses one
    // channel to its smallest value -- which is why the call parser picks a
    // definition by argument count rather than by name alone.
    {"min", 1, S_MIN},          // min(Y)              smallest sample
    {"max", 1, S_MAX},          // max(Y)              largest sample
    {"mean", 1, S_MEAN},        // mean(Y)             arithmetic mean
};

bool ieq(const std::string& a, const char* b) {
    std::size_t i = 0;
    for (; i < a.size() && b[i]; ++i) {
        char x = a[i], y = b[i];
        if (x >= 'A' && x <= 'Z') x = static_cast<char>(x + 32);
        if (y >= 'A' && y <= 'Z') y = static_cast<char>(y + 32);
        if (x != y) return false;
    }
    return i == a.size() && b[i] == '\0';
}

bool ident_char(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_' || c == '.' || c == ':';
}

bool ident_start(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

float apply_fn1(std::uint8_t id, float v) {
    switch (id) {
        case F_ABS:   return std::fabs(v);
        case F_SQRT:  return v < 0.0f ? kNaN : std::sqrt(v);
        case F_SIN:   return std::sin(v);
        case F_COS:   return std::cos(v);
        case F_TAN:   return std::tan(v);
        case F_ASIN:  return (v < -1.0f || v > 1.0f) ? kNaN : std::asin(v);
        case F_ACOS:  return (v < -1.0f || v > 1.0f) ? kNaN : std::acos(v);
        case F_ATAN:  return std::atan(v);
        case F_EXP:   return std::exp(v);
        case F_LOG:   return v <= 0.0f ? kNaN : std::log(v);
        case F_LOG10: return v <= 0.0f ? kNaN : std::log10(v);
        case F_FLOOR: return std::floor(v);
        case F_CEIL:  return std::ceil(v);
        case F_ROUND: return std::round(v);
        case F_SIGN:  return (v > 0.0f) - (v < 0.0f);
        case F_DEG:   return v * 57.29577951308232f;
        case F_RAD:   return v * 0.017453292519943295f;
        default:      return kNaN;
    }
}

float apply_fn2(std::uint8_t id, float a, float b) {
    switch (id) {
        case F_MIN:   return (std::isnan(a) || std::isnan(b)) ? kNaN : (a < b ? a : b);
        case F_MAX:   return (std::isnan(a) || std::isnan(b)) ? kNaN : (a > b ? a : b);
        case F_ATAN2: return std::atan2(a, b);
        case F_POW:   return std::pow(a, b);
        case F_MOD:   return b == 0.0f ? kNaN : std::fmod(a, b);
        case F_HYPOT: return std::hypot(a, b);
        default:      return kNaN;
    }
}

float apply_fn3(std::uint8_t id, float a, float b, float c) {
    switch (id) {
        case F_CLAMP: return std::isnan(a) ? kNaN : (a < b ? b : (a > c ? c : a));
        case F_IF:    return (a != 0.0f && !std::isnan(a)) ? b : c;
        default:      return kNaN;
    }
}

constexpr double kTwoPi = 6.283185307179586;

// dY/dX by central differences, one-sided at the ends. A zero step yields NaN
// rather than an infinity, for the same reason division does elsewhere: an
// infinity would destroy the plot's y-range, a NaN just breaks the line.
void apply_derivative(float* y, const float* x, std::size_t n) {
    if (n < 2) {
        if (n == 1) y[0] = 0.0f;
        return;
    }
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t lo = (i == 0) ? 0 : i - 1;
        const std::size_t hi = (i + 1 < n) ? i + 1 : n - 1;
        const double dx = static_cast<double>(x[hi]) - static_cast<double>(x[lo]);
        out[i] = (dx == 0.0)
                     ? kNaN
                     : static_cast<float>((static_cast<double>(y[hi]) -
                                           static_cast<double>(y[lo])) / dx);
    }
    std::memcpy(y, out.data(), n * sizeof(float));
}

// Cumulative trapezoidal integral, starting at zero.
void apply_integral(float* y, const float* x, std::size_t n) {
    if (n == 0) return;
    double total = 0.0;
    double previous_y = static_cast<double>(y[0]);
    double previous_x = static_cast<double>(x[0]);
    y[0] = 0.0f;
    for (std::size_t i = 1; i < n; ++i) {
        const double value = static_cast<double>(y[i]);
        const double position = static_cast<double>(x[i]);
        const double step = position - previous_x;
        if (!std::isnan(value) && !std::isnan(previous_y) && !std::isnan(step))
            total += 0.5 * (value + previous_y) * step;
        previous_y = value;
        previous_x = position;
        y[i] = static_cast<float>(total);
    }
}

// First-order IIR filters, with the step taken from the x channel at every
// sample rather than assumed constant -- AOE output is not always uniformly
// sampled, and assuming it is silently changes the cutoff.
void apply_lowpass(float* y, const float* x, std::size_t n, double cutoff_hz) {
    if (n == 0) return;
    if (!(cutoff_hz > 0.0)) return;  // no filtering rather than a divide by zero

    const double rc = 1.0 / (kTwoPi * cutoff_hz);
    double state = static_cast<double>(y[0]);
    for (std::size_t i = 1; i < n; ++i) {
        const double step = static_cast<double>(x[i]) - static_cast<double>(x[i - 1]);
        const double alpha = (step > 0.0) ? step / (rc + step) : 0.0;
        const double value = static_cast<double>(y[i]);
        if (!std::isnan(value)) state += alpha * (value - state);
        y[i] = static_cast<float>(state);
    }
}

void apply_highpass(float* y, const float* x, std::size_t n, double cutoff_hz) {
    if (n == 0) return;
    if (!(cutoff_hz > 0.0)) return;

    const double rc = 1.0 / (kTwoPi * cutoff_hz);
    double previous_input = static_cast<double>(y[0]);
    double state = 0.0;
    y[0] = 0.0f;
    for (std::size_t i = 1; i < n; ++i) {
        const double step = static_cast<double>(x[i]) - static_cast<double>(x[i - 1]);
        const double alpha = (step > 0.0) ? rc / (rc + step) : 1.0;
        const double value = static_cast<double>(y[i]);
        state = alpha * (state + value - previous_input);
        previous_input = value;
        y[i] = static_cast<float>(state);
    }
}

// Centred moving average over `width` samples, via a running sum so the cost is
// independent of the window size.
void apply_moving_average(float* y, std::size_t n, int width) {
    if (n == 0 || width <= 1) return;
    if (static_cast<std::size_t>(width) > n) width = static_cast<int>(n);

    const int half = width / 2;
    std::vector<double> prefix(n + 1, 0.0);
    std::vector<std::size_t> counts(n + 1, 0);
    for (std::size_t i = 0; i < n; ++i) {
        const bool usable = !std::isnan(y[i]);
        prefix[i + 1] = prefix[i] + (usable ? static_cast<double>(y[i]) : 0.0);
        counts[i + 1] = counts[i] + (usable ? 1u : 0u);
    }

    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t lo = (static_cast<std::size_t>(half) > i) ? 0 : i - half;
        std::size_t hi = i + static_cast<std::size_t>(width - half);
        if (hi > n) hi = n;
        const std::size_t used = counts[hi] - counts[lo];
        out[i] = used ? static_cast<float>((prefix[hi] - prefix[lo]) /
                                           static_cast<double>(used))
                      : kNaN;
    }
    std::memcpy(y, out.data(), n * sizeof(float));
}

void apply_delta(float* y, std::size_t n) {
    if (n == 0) return;
    const float base = y[0];
    for (std::size_t i = 0; i < n; ++i) y[i] -= base;
}

}  // namespace

float Formula::binary_apply(Op op, float a, float b) {
    switch (op) {
        case Op::Add: return a + b;
        case Op::Sub: return a - b;
        case Op::Mul: return a * b;
        // NaN rather than +/-inf: an infinity silently destroys the plot's
        // y-range, whereas a NaN simply breaks the line where the data is bad.
        case Op::Div: return b == 0.0f ? kNaN : a / b;
        case Op::Mod: return b == 0.0f ? kNaN : std::fmod(a, b);
        case Op::Pow: return std::pow(a, b);
        case Op::Lt:  return a < b ? 1.0f : 0.0f;
        case Op::Gt:  return a > b ? 1.0f : 0.0f;
        case Op::Le:  return a <= b ? 1.0f : 0.0f;
        case Op::Ge:  return a >= b ? 1.0f : 0.0f;
        case Op::Eq:  return a == b ? 1.0f : 0.0f;
        case Op::Ne:  return a != b ? 1.0f : 0.0f;
        case Op::And: return (a != 0.0f && b != 0.0f) ? 1.0f : 0.0f;
        case Op::Or:  return (a != 0.0f || b != 0.0f) ? 1.0f : 0.0f;
        default:      return kNaN;
    }
}

// ------------------------------------------------------------- Parser

class Parser {
public:
    Parser(const std::string& text, Formula& out, std::string& error)
        : p_(text.c_str()), begin_(text.c_str()), out_(out), error_(error) {}

    bool run() {
        skip();
        if (*p_ == '\0') {
            fail("empty expression");
            return false;
        }
        parse_or();
        if (!error_.empty()) return false;
        skip();
        if (*p_ != '\0') {
            fail("unexpected trailing text");
            return false;
        }
        return true;
    }

private:
    using Op = Formula::Op;
    using Instr = Formula::Instr;

    void emit(Op op) {
        Instr in;
        in.op = op;
        out_.code_.push_back(in);
    }
    void emit_const(float v) {
        Instr in;
        in.op = Op::Const;
        in.value = v;
        out_.code_.push_back(in);
    }
    void emit_var(int slot) {
        Instr in;
        in.op = Op::Var;
        in.a = slot;
        out_.code_.push_back(in);
    }
    void emit_fn(Op op, std::uint8_t id) {
        Instr in;
        in.op = op;
        in.fn = id;
        out_.code_.push_back(in);
    }

    void fail(const std::string& message) {
        if (!error_.empty()) return;
        const std::size_t pos = static_cast<std::size_t>(p_ - begin_);
        error_ = message + " (at position " + std::to_string(pos) + ")";
    }

    void skip() {
        while (*p_ == ' ' || *p_ == '\t' || *p_ == '\r' || *p_ == '\n') ++p_;
    }

    bool match(const char* token) {
        skip();
        const std::size_t n = std::strlen(token);
        if (std::strncmp(p_, token, n) != 0) return false;
        // Do not let "<" swallow the "<=" case, or "|" the "||".
        p_ += n;
        return true;
    }

    int variable_slot(const std::string& name) {
        for (std::size_t i = 0; i < out_.variables_.size(); ++i)
            if (out_.variables_[i] == name) return static_cast<int>(i);
        out_.variables_.push_back(name);
        return static_cast<int>(out_.variables_.size() - 1);
    }

    std::string read_identifier() {
        const char* s = p_;
        while (ident_char(*p_)) ++p_;
        return std::string(s, static_cast<std::size_t>(p_ - s));
    }

    // or := and ( '||' and )*
    void parse_or() {
        parse_and();
        for (;;) {
            skip();
            if (p_[0] == '|' && p_[1] == '|') {
                p_ += 2;
                parse_and();
                emit(Op::Or);
            } else {
                return;
            }
        }
    }

    void parse_and() {
        parse_equality();
        for (;;) {
            skip();
            if (p_[0] == '&' && p_[1] == '&') {
                p_ += 2;
                parse_equality();
                emit(Op::And);
            } else {
                return;
            }
        }
    }

    void parse_equality() {
        parse_relational();
        for (;;) {
            skip();
            if (p_[0] == '=' && p_[1] == '=') {
                p_ += 2;
                parse_relational();
                emit(Op::Eq);
            } else if (p_[0] == '!' && p_[1] == '=') {
                p_ += 2;
                parse_relational();
                emit(Op::Ne);
            } else {
                return;
            }
        }
    }

    void parse_relational() {
        parse_additive();
        for (;;) {
            skip();
            if (p_[0] == '<' && p_[1] == '=') {
                p_ += 2;
                parse_additive();
                emit(Op::Le);
            } else if (p_[0] == '>' && p_[1] == '=') {
                p_ += 2;
                parse_additive();
                emit(Op::Ge);
            } else if (p_[0] == '<') {
                ++p_;
                parse_additive();
                emit(Op::Lt);
            } else if (p_[0] == '>') {
                ++p_;
                parse_additive();
                emit(Op::Gt);
            } else {
                return;
            }
        }
    }

    void parse_additive() {
        parse_term();
        for (;;) {
            skip();
            if (*p_ == '+') {
                ++p_;
                parse_term();
                emit(Op::Add);
            } else if (*p_ == '-') {
                ++p_;
                parse_term();
                emit(Op::Sub);
            } else {
                return;
            }
        }
    }

    void parse_term() {
        parse_unary();
        for (;;) {
            skip();
            if (*p_ == '*') {
                ++p_;
                parse_unary();
                emit(Op::Mul);
            } else if (*p_ == '/') {
                ++p_;
                parse_unary();
                emit(Op::Div);
            } else if (*p_ == '%') {
                ++p_;
                parse_unary();
                emit(Op::Mod);
            } else {
                return;
            }
        }
    }

    void parse_unary() {
        skip();
        if (*p_ == '-') {
            ++p_;
            parse_unary();
            emit(Op::Neg);
            return;
        }
        if (*p_ == '+') {
            ++p_;
            parse_unary();
            return;
        }
        if (*p_ == '!' && p_[1] != '=') {
            ++p_;
            parse_unary();
            emit(Op::Not);
            return;
        }
        parse_power();
    }

    // Right-associative, and binds tighter than unary minus on the left:
    // -2^2 is -(2^2), matching every engineering calculator.
    void parse_power() {
        parse_primary();
        skip();
        if (*p_ == '^') {
            ++p_;
            parse_unary();
            emit(Op::Pow);
        }
    }

    void parse_primary() {
        skip();
        if (error_.size()) return;

        if (*p_ == '(') {
            ++p_;
            parse_or();
            skip();
            if (*p_ != ')') {
                fail("expected ')'");
                return;
            }
            ++p_;
            return;
        }

        if ((*p_ >= '0' && *p_ <= '9') || (*p_ == '.' && p_[1] >= '0' && p_[1] <= '9')) {
            char* end = nullptr;
            const double v = std::strtod(p_, &end);
            if (end == p_) {
                fail("malformed number");
                return;
            }
            p_ = end;
            emit_const(static_cast<float>(v));
            return;
        }

        if (ident_start(*p_)) {
            const std::string name = read_identifier();
            skip();

            if (*p_ == '(') return parse_call(name);
            if (*p_ == '[') return parse_lookup(name);

            if (ieq(name, "pi")) {
                emit_const(3.14159265358979323846f);
                return;
            }
            if (ieq(name, "e")) {
                emit_const(2.71828182845904523536f);
                return;
            }
            if (ieq(name, "nan")) {
                emit_const(kNaN);
                return;
            }

            emit_var(variable_slot(name));
            return;
        }

        fail("unexpected character");
    }

    void parse_call(const std::string& name) {
        ++p_;  // consume '('

        std::vector<int> arg_marks;
        skip();
        if (*p_ == ')') {
            ++p_;
        } else {
            for (;;) {
                parse_or();
                if (!error_.empty()) return;
                arg_marks.push_back(0);
                skip();
                if (*p_ == ',') {
                    ++p_;
                    continue;
                }
                if (*p_ == ')') {
                    ++p_;
                    break;
                }
                fail("expected ',' or ')' in function arguments");
                return;
            }
        }
        const int argc = static_cast<int>(arg_marks.size());

        // A name can be defined at more than one arity -- min and max are both
        // pointwise over two channels and a reduction over one -- so the
        // argument count chooses which definition is meant, and the error
        // message lists the counts that would have worked.
        std::vector<int> accepted;

        for (const auto& fn : kFn1)
            if (ieq(name, fn.name)) {
                if (argc == 1) {
                    emit_fn(Op::Fn1, fn.id);
                    return;
                }
                accepted.push_back(1);
            }
        for (const auto& fn : kFn2)
            if (ieq(name, fn.name)) {
                if (argc == 2) {
                    emit_fn(Op::Fn2, fn.id);
                    return;
                }
                accepted.push_back(2);
            }
        for (const auto& fn : kFn3)
            if (ieq(name, fn.name)) {
                if (argc == 3) {
                    emit_fn(Op::Fn3, fn.id);
                    return;
                }
                accepted.push_back(3);
            }
        for (const auto& fn : kFnSeries)
            if (ieq(name, fn.name)) {
                if (argc == fn.arity) {
                    Instr in;
                    in.op = Op::Series;
                    in.fn = fn.id;
                    in.a = fn.arity;
                    out_.code_.push_back(in);
                    return;
                }
                accepted.push_back(fn.arity);
            }

        if (accepted.empty()) {
            fail("unknown function '" + name + "'");
            return;
        }

        std::sort(accepted.begin(), accepted.end());
        accepted.erase(std::unique(accepted.begin(), accepted.end()), accepted.end());
        std::string counts;
        for (std::size_t i = 0; i < accepted.size(); ++i) {
            if (i) counts += (i + 1 == accepted.size()) ? " or " : ", ";
            counts += std::to_string(accepted[i]);
        }
        fail(name + "() takes " + counts + " argument" +
             ((accepted.size() == 1 && accepted[0] == 1) ? "" : "s") + ", not " +
             std::to_string(argc));
    }

    // NAME[COND=value] -- the value of NAME at the sample where COND is
    // closest to `value`. Evaluates to a scalar broadcast over the series.
    void parse_lookup(const std::string& name) {
        ++p_;  // consume '['
        skip();
        if (!ident_start(*p_)) {
            fail("expected a parameter name inside [ ]");
            return;
        }
        const std::string cond = read_identifier();
        skip();
        if (*p_ != '=') {
            fail("expected '=' in lookup, e.g. SPEED[TIME=0.5]");
            return;
        }
        ++p_;
        skip();

        // The value may be a number, or one of the two ends of the run.
        // NZ[TIME=first] and NZ[TIME=last] are what an engineer reaches for to
        // normalise against the start or the end of a manoeuvre, and writing
        // them as numbers is impossible when every run has a different length.
        std::uint8_t mode = kLookupNearest;
        float target = 0.0f;

        if (ident_start(*p_)) {
            const std::string keyword = read_identifier();
            if (ieq(keyword, "first")) {
                mode = kLookupFirst;
            } else if (ieq(keyword, "last")) {
                mode = kLookupLast;
            } else {
                fail("expected a number, 'first' or 'last' in lookup, "
                     "e.g. SPEED[TIME=0.5] or SPEED[TIME=last]");
                return;
            }
        } else {
            char* end = nullptr;
            const double v = std::strtod(p_, &end);
            if (end == p_) {
                fail("expected a number, 'first' or 'last' in lookup");
                return;
            }
            p_ = end;
            target = static_cast<float>(v);
        }

        skip();
        if (*p_ != ']') {
            fail("expected ']' in lookup");
            return;
        }
        ++p_;

        Instr in;
        in.op = Op::Lookup;
        in.fn = mode;
        in.a = variable_slot(name);
        in.b = variable_slot(cond);
        in.value = target;
        out_.code_.push_back(in);
    }

    const char* p_;
    const char* begin_;
    Formula& out_;
    std::string& error_;
};

// ---------------------------------------------------------- compile

Formula Formula::compile(const std::string& expression, std::string* error) {
    Formula formula;
    formula.source_ = expression;

    std::string local_error;
    Parser parser(expression, formula, local_error);
    const bool ok = parser.run();

    if (!ok || !local_error.empty()) {
        if (error) *error = local_error.empty() ? "invalid expression" : local_error;
        formula.valid_ = false;
        formula.code_.clear();
        return formula;
    }

    if (error) error->clear();
    formula.valid_ = true;
    return formula;
}

bool Formula::validate(const std::string& expression,
                       const std::function<bool(const std::string&)>& known,
                       std::string* error) {
    std::string local;
    Formula formula = compile(expression, &local);
    if (!formula.valid()) {
        if (error) *error = local;
        return false;
    }
    if (known) {
        for (const auto& name : formula.variables()) {
            if (!known(name)) {
                if (error) *error = "unknown parameter '" + name + "'";
                return false;
            }
        }
    }
    if (error) error->clear();
    return true;
}

// --------------------------------------------------------- evaluate

namespace {

// A value on the evaluation stack: either a single float broadcast over the
// series, or a scratch buffer holding one float per sample. Keeping scalars
// scalar is what stops `X * 2 + 1` from allocating three full-length buffers.
struct Value {
    bool scalar = true;
    float s = 0.0f;
    int buf = -1;  // index into the scratch pool when !scalar
};

// Pool of same-length float buffers, reused across the whole expression so a
// deep formula allocates a handful of buffers rather than one per operator.
// Growing pool_ is safe for pointers already handed out: reallocating a
// vector<vector<float>> moves the inner vectors, and a moved vector keeps its
// heap buffer, so previously returned float* stay valid.
class Scratch {
public:
    explicit Scratch(std::size_t n) : n_(n) {}

    int acquire() {
        if (!free_.empty()) {
            const int idx = free_.back();
            free_.pop_back();
            return idx;
        }
        pool_.emplace_back(n_);
        return static_cast<int>(pool_.size()) - 1;
    }
    void release(int idx) {
        if (idx >= 0) free_.push_back(idx);
    }
    float* data(int idx) { return pool_[static_cast<std::size_t>(idx)].data(); }

private:
    std::size_t n_;
    std::vector<std::vector<float>> pool_;
    std::vector<int> free_;
};

}  // namespace

bool Formula::evaluate(const ChannelResolver& resolver, std::size_t count,
                       std::vector<float>& out, std::string* error) const {
    if (!valid_) {
        if (error) *error = "formula did not compile";
        return false;
    }

    // Bind every variable once.
    std::vector<const float*> bound(variables_.size(), nullptr);
    std::vector<std::size_t> lengths(variables_.size(), 0);
    std::size_t shortest = 0;
    bool have_series = false;

    for (std::size_t i = 0; i < variables_.size(); ++i) {
        std::size_t n = 0;
        const float* data = resolver ? resolver(variables_[i], n) : nullptr;
        if (!data || n == 0) {
            if (error) *error = "no data for parameter '" + variables_[i] + "'";
            return false;
        }
        bound[i] = data;
        lengths[i] = n;
        if (!have_series || n < shortest) {
            shortest = n;
            have_series = true;
        }
    }

    std::size_t n = count;
    if (n == 0) n = have_series ? shortest : 0;
    if (n == 0) {
        if (error) *error = "cannot determine sample count for a constant formula";
        return false;
    }
    // Never read past the shortest channel involved.
    if (have_series && n > shortest) n = shortest;

    out.assign(n, 0.0f);

    Scratch scratch(n);
    std::vector<Value> stack;
    stack.reserve(code_.size() + 1);

    auto materialise = [&](Value& v) -> float* {
        if (!v.scalar) return scratch.data(v.buf);
        const int idx = scratch.acquire();
        float* dst = scratch.data(idx);
        for (std::size_t i = 0; i < n; ++i) dst[i] = v.s;
        v.scalar = false;
        v.buf = idx;
        return dst;
    };

    for (const Instr& in : code_) {
        switch (in.op) {
            case Op::Const: {
                Value v;
                v.scalar = true;
                v.s = in.value;
                stack.push_back(v);
                break;
            }
            case Op::Var: {
                const float* src = bound[static_cast<std::size_t>(in.a)];
                const int idx = scratch.acquire();
                float* dst = scratch.data(idx);
                std::memcpy(dst, src, n * sizeof(float));
                Value v;
                v.scalar = false;
                v.buf = idx;
                stack.push_back(v);
                break;
            }
            case Op::Lookup: {
                const float* value_src = bound[static_cast<std::size_t>(in.a)];
                const float* cond_src = bound[static_cast<std::size_t>(in.b)];
                const std::size_t limit =
                    std::min(lengths[static_cast<std::size_t>(in.a)],
                             lengths[static_cast<std::size_t>(in.b)]);

                std::size_t best = 0;
                bool found = false;

                if (in.fn == kLookupFirst || in.fn == kLookupLast) {
                    // The ends of the run, skipping gaps at either end so a
                    // channel that starts or finishes with NaN still answers
                    // with a real sample rather than with the gap.
                    if (in.fn == kLookupFirst) {
                        for (std::size_t i = 0; i < limit; ++i)
                            if (!std::isnan(cond_src[i]) && !std::isnan(value_src[i])) {
                                best = i;
                                found = true;
                                break;
                            }
                    } else {
                        for (std::size_t i = limit; i > 0; --i)
                            if (!std::isnan(cond_src[i - 1]) &&
                                !std::isnan(value_src[i - 1])) {
                                best = i - 1;
                                found = true;
                                break;
                            }
                    }
                } else {
                    float best_diff = std::numeric_limits<float>::max();
                    for (std::size_t i = 0; i < limit; ++i) {
                        const float d = std::fabs(cond_src[i] - in.value);
                        if (d < best_diff) {
                            best_diff = d;
                            best = i;
                            found = true;
                        }
                    }
                }

                Value v;
                v.scalar = true;
                v.s = found ? value_src[best] : kNaN;
                stack.push_back(v);
                break;
            }
            case Op::Neg:
            case Op::Not: {
                if (stack.empty()) break;
                Value& v = stack.back();
                if (v.scalar) {
                    v.s = (in.op == Op::Neg) ? -v.s
                                             : ((v.s == 0.0f) ? 1.0f : 0.0f);
                } else {
                    float* d = scratch.data(v.buf);
                    if (in.op == Op::Neg)
                        for (std::size_t i = 0; i < n; ++i) d[i] = -d[i];
                    else
                        for (std::size_t i = 0; i < n; ++i)
                            d[i] = (d[i] == 0.0f) ? 1.0f : 0.0f;
                }
                break;
            }
            case Op::Fn1: {
                if (stack.empty()) break;
                Value& v = stack.back();
                if (v.scalar) {
                    v.s = apply_fn1(in.fn, v.s);
                } else {
                    float* d = scratch.data(v.buf);
                    for (std::size_t i = 0; i < n; ++i) d[i] = apply_fn1(in.fn, d[i]);
                }
                break;
            }
            case Op::Fn2: {
                if (stack.size() < 2) break;
                Value rhs = stack.back();
                stack.pop_back();
                Value& lhs = stack.back();
                if (lhs.scalar && rhs.scalar) {
                    lhs.s = apply_fn2(in.fn, lhs.s, rhs.s);
                } else {
                    float* d = materialise(lhs);
                    if (rhs.scalar) {
                        for (std::size_t i = 0; i < n; ++i)
                            d[i] = apply_fn2(in.fn, d[i], rhs.s);
                    } else {
                        const float* r = scratch.data(rhs.buf);
                        for (std::size_t i = 0; i < n; ++i)
                            d[i] = apply_fn2(in.fn, d[i], r[i]);
                    }
                }
                scratch.release(rhs.scalar ? -1 : rhs.buf);
                break;
            }
            case Op::Fn3: {
                if (stack.size() < 3) break;
                Value c = stack.back();
                stack.pop_back();
                Value b = stack.back();
                stack.pop_back();
                Value& a = stack.back();
                if (a.scalar && b.scalar && c.scalar) {
                    a.s = apply_fn3(in.fn, a.s, b.s, c.s);
                } else {
                    float* d = materialise(a);
                    const float* bp = b.scalar ? nullptr : scratch.data(b.buf);
                    const float* cp = c.scalar ? nullptr : scratch.data(c.buf);
                    for (std::size_t i = 0; i < n; ++i)
                        d[i] = apply_fn3(in.fn, d[i], bp ? bp[i] : b.s,
                                         cp ? cp[i] : c.s);
                }
                scratch.release(b.scalar ? -1 : b.buf);
                scratch.release(c.scalar ? -1 : c.buf);
                break;
            }
            case Op::Series: {
                // Whole-series operations. The value argument is always
                // materialised because these read neighbours, and the result
                // replaces it in place.
                const int argc = in.a;
                if (static_cast<int>(stack.size()) < argc) break;

                Value third, second;
                if (argc >= 3) {
                    third = stack.back();
                    stack.pop_back();
                }
                if (argc >= 2) {
                    second = stack.back();
                    stack.pop_back();
                }
                Value& first = stack.back();
                float* values = materialise(first);

                switch (in.fn) {
                    case S_DERIV:
                    case S_INTEG: {
                        // The x argument must be a real series; a constant x
                        // would make every step zero.
                        if (second.scalar) {
                            for (std::size_t i = 0; i < n; ++i) values[i] = kNaN;
                            break;
                        }
                        const float* axis = scratch.data(second.buf);
                        if (in.fn == S_DERIV)
                            apply_derivative(values, axis, n);
                        else
                            apply_integral(values, axis, n);
                        break;
                    }
                    case S_LOWPASS:
                    case S_HIGHPASS: {
                        if (second.scalar) {
                            for (std::size_t i = 0; i < n; ++i) values[i] = kNaN;
                            break;
                        }
                        const float* axis = scratch.data(second.buf);
                        const double cutoff =
                            third.scalar ? static_cast<double>(third.s)
                                         : static_cast<double>(scratch.data(third.buf)[0]);
                        if (in.fn == S_LOWPASS)
                            apply_lowpass(values, axis, n, cutoff);
                        else
                            apply_highpass(values, axis, n, cutoff);
                        break;
                    }
                    case S_MOVAVG: {
                        const double width =
                            second.scalar ? static_cast<double>(second.s)
                                          : static_cast<double>(scratch.data(second.buf)[0]);
                        apply_moving_average(values, n, static_cast<int>(width));
                        break;
                    }
                    case S_DELTA:
                        apply_delta(values, n);
                        break;
                    case S_MIN:
                    case S_MAX:
                    case S_MEAN: {
                        // A reduction: one number, held flat across the run, so
                        // it can be combined with per-sample channels by the
                        // ordinary arithmetic that follows.
                        double lo = 0.0, hi = 0.0, sum = 0.0;
                        std::size_t seen = 0;
                        for (std::size_t i = 0; i < n; ++i) {
                            const float v = values[i];
                            if (std::isnan(v) || std::isinf(v)) continue;
                            if (seen == 0) {
                                lo = hi = static_cast<double>(v);
                            } else {
                                if (v < lo) lo = v;
                                if (v > hi) hi = v;
                            }
                            sum += static_cast<double>(v);
                            ++seen;
                        }
                        float answer = kNaN;
                        if (seen > 0) {
                            if (in.fn == S_MIN)
                                answer = static_cast<float>(lo);
                            else if (in.fn == S_MAX)
                                answer = static_cast<float>(hi);
                            else
                                answer = static_cast<float>(sum /
                                                            static_cast<double>(seen));
                        }
                        for (std::size_t i = 0; i < n; ++i) values[i] = answer;
                        break;
                    }
                    default:
                        for (std::size_t i = 0; i < n; ++i) values[i] = kNaN;
                        break;
                }

                scratch.release(second.scalar ? -1 : second.buf);
                scratch.release(third.scalar ? -1 : third.buf);
                break;
            }
            default: {  // binary arithmetic / comparison / logic
                if (stack.size() < 2) break;
                Value rhs = stack.back();
                stack.pop_back();
                Value& lhs = stack.back();

                if (lhs.scalar && rhs.scalar) {
                    lhs.s = Formula::binary_apply(in.op, lhs.s, rhs.s);
                } else {
                    float* d = materialise(lhs);
                    if (rhs.scalar) {
                        const float r = rhs.s;
                        for (std::size_t i = 0; i < n; ++i)
                            d[i] = Formula::binary_apply(in.op, d[i], r);
                    } else {
                        const float* r = scratch.data(rhs.buf);
                        for (std::size_t i = 0; i < n; ++i)
                            d[i] = Formula::binary_apply(in.op, d[i], r[i]);
                    }
                }
                scratch.release(rhs.scalar ? -1 : rhs.buf);
                break;
            }
        }
    }

    if (stack.empty()) {
        if (error) *error = "expression produced no value";
        return false;
    }

    const Value& result = stack.back();
    if (result.scalar) {
        std::fill(out.begin(), out.end(), result.s);
    } else {
        std::memcpy(out.data(), scratch.data(result.buf), n * sizeof(float));
    }

    if (error) error->clear();
    return true;
}

}  // namespace aoe
