module Plotting

using Random
using Statistics
using DataFrames
using Plots

# Function to create a histogram
function create_histogram(data, column, title)
    histogram(data[column], title=title)
end

# Function to create a scatter plot
function create_scatter(data, x_column, y_column, title)
    scatter(data[x_column], data[y_column], title=title)
end

# Function to create a heatmap
function create_heatmap(xlabels, ylabels, matrix, title)
    heatmap(xlabels, ylabels, matrix, title=title, show_text=true)
end

# Function to create a line plot
function create_line(data, x_column, y_column, title)
    plot(data[x_column], data[y_column], title=title)
end

function custom_plots(program::Int, samples)
    plot_log = Dict{String, Any}()

    if program == 1 || program == 6 || program == 11
        data = DataFrame([j => sample for (j, sample) in enumerate(samples)], [:sample, :mu])
        plot_log["Program $program"] = create_histogram(data, :mu, "Program $program; mu")
    elseif program == 2 || program == 7 || program == 12
        data = DataFrame([j => vcat(parts...) for (j, parts) in enumerate(samples)], [:sample, :slope, :bias])
        plot_log["Program $program; slope"] = create_histogram(data, :slope, "Program $program; slope")
        plot_log["Program $program; bias"] = create_histogram(data, :bias, "Program $program; bias")
        plot_log["Program $program; scatter"] = create_scatter(data, :slope, :bias, "Program $program; slope vs. bias")
    elseif program == 3 || program == 8 || program == 13
        data = convert(Matrix, samples)
        xs = range(0, length(data[1, :]) - 1, length=length(data[1, :]))
        x = Float64[]; y = Float64[]
        for i in 1:size(data, 1)
            for j in 1:size(data, 2)
                push!(x, xs[j])
                push!(y, data[i, j])
            end
        end
        xedges = range(-0.5, size(data, 2) - 0.5, length=size(data, 2) + 1)
        yedges = range(-0.5, maximum(data) + 0.5, length=Int(maximum(data)) + 2)
        matrix, _, _ = histogram2d(x, y, bins=(xedges, yedges))
        xlabels = xedges[1:end-1] .+ 0.5; ylabels = yedges[1:end-1] .+ 0.5
        plot_log["Program $program; heatmap"] = create_heatmap(xlabels, ylabels, matrix', "Program $program; heatmap")
    elseif program == 4 || program == 9 || program == 14
        x_values = 0:size(samples, 2) - 1
        for (y_values, name) in zip([mean(samples, dims=1), std(samples, dims=1)], ["mean", "std"])
            data = DataFrame([x_values => y_values], [:position, Symbol(name)])
            title = "Program $program; $name"
            plot_log[title] = create_line(data, :position, Symbol(name), title)
        end
    elseif program == 5 || program == 10
        data = DataFrame([j => sample for (j, sample) in enumerate(samples)], [:sample, :x])
        plot_log["Program $program"] = create_histogram(data, :x, "Program $program; Is it raining?")
    else
        error("Program not recognized")
    end

    println(plot_log)  # Placeholder for actual logging
end

end # module Plotting